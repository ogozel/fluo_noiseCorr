# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:46:09 2021

@author: Olivia Gozel

Analyze the data using home-made functions
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze
import functions_preprocess

dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data\\'



#%% Parameters of the preprocessed data to analyze

filepath = dataDir + 'L4_cytosolic_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,'L4_cytosolic_dataSpecs')

dataType = 'L4_cytosolic'
idxDataset = 0 # 0,1,4,8: nice corr=f(dist); 2,3,5,6,7,9: bad corr=f(dist)
dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor


#%% Load data

filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist2d5um.hdf'

fluo = pd.read_hdf(filepath,'fluo')
charROI = pd.read_hdf(filepath,'charROI')
charTrials = pd.read_hdf(filepath,'charTrials')
positionROI = pd.read_hdf(filepath,'positionROI')
distROI = pd.read_hdf(filepath,'distROI')
fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()

if 'pupilArea' in charTrials:
    bool_pupil = True

store = pd.HDFStore(filepath,mode='r')
if 'motAvg' in store:
    bool_motion = True
    motAvg = pd.read_hdf(filepath,'motAvg')
    uMotMask = pd.read_hdf(filepath,'uMotMask')

### !!!
#fluo = fluo[0:13000]
#charTrials = charTrials[0:13000]
### !!!


#%% Plot ROIs with their selectivity

functions_preprocess.plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)



#%% Plot the pairwise correlation as a function of pairwise distance

idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
thisDistROI = np.array(distROI)[idxOS,:][:,idxOS]

# Evoked frames (OS neurons only)
idxStim = np.squeeze(np.array(np.where(charTrials['FrameType']=='Stimulus')))
thisFluo = np.array(fluo)[idxStim,:][:,idxOS]

binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,startBin=7.5,binSize=10,title='Evoked - OS only')


# Spontaneous frames (OS neurons only)
idxBlank = np.squeeze(np.array(np.where(charTrials['FrameType']=='Blank')))
thisFluo = np.array(fluo)[idxBlank,:][:,idxOS]

functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,startBin=2.5,binSize=10,title='Spontaneous - OS only')



#%% Effect of the number of trials

# Evoked frames (OS neurons only)
idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
idxStim45 = np.squeeze(np.array(np.where((charTrials['FrameType']=='Stimulus')&(charTrials['Orientation']==globalParams.ori[0]))))
thisFluo = np.array(fluo)[idxStim45,:][:,idxOS]
functions_analyze.plot_ftrials_neuralPCs(thisFluo,nFramesPerTrial=globalParams.nStimFrames,title='Evoked (45°) - OS only',nInstances=50)

# Spontaneous frames (OS neurons only)
idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
idxBlank = np.squeeze(np.array(np.where(charTrials['FrameType']=='Blank')))
thisFluo = np.array(fluo)[idxBlank,:][:,idxOS]
# NB: keep nFramesPerTrial=globalParams.nStimFrames because we need enough frames to estimate PCs
functions_analyze.plot_ftrials_neuralPCs(thisFluo,nFramesPerTrial=globalParams.nStimFrames,title='Spontaneous - OS only',nInstances=50)



#%% Effect of the number of trials for the estimation of the pairwise correlations as a function of pairwise distance

idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
thisDistROI = np.array(distROI)[idxOS,:][:,idxOS]

# Evoked frames (OS neurons only)
idxStim45 = np.squeeze(np.array(np.where((charTrials['FrameType']=='Stimulus')&(charTrials['Orientation']==globalParams.ori[0]))))
thisFluo = np.array(fluo)[idxStim45,:][:,idxOS]

functions_analyze.plot_ftrials_corr_fdist(thisFluo,thisDistROI,nFramesPerTrial=globalParams.nStimFrames,binSize=10,nInstances=50,title='Evoked (45°) - OS only')


# Spontaneous frames (OS neurons only)
idxBlank = np.squeeze(np.array(np.where(charTrials['FrameType']=='Blank')))
thisFluo = np.array(fluo)[idxBlank,:][:,idxOS]

functions_analyze.plot_ftrials_corr_fdist(thisFluo,thisDistROI,nFramesPerTrial=globalParams.nBlankFrames,binSize=10,nInstances=50,title='Spontaneous - OS only')




#%% Compute the neural PCs

# Select the ROIs we are interested in
idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
#thisFluo = np.array(fluo)
thisFluo = np.array(fluo)[:,idxOS]

# All frames, only OS neurons
neuralProj, percVarExpl = functions_analyze.compute_neuralPCs(thisFluo,bool_plot=1,title='All frames - OS only')




#%% Define the traces of interest

traceNeural = neuralProj[:,0]
tracePupil = np.array(charTrials['pupilArea']).astype(float)
traceMotion = np.array(charTrials['motSVD1'])

# Flip trace if average is negative
if np.mean(traceMotion) < 0:
    traceMotion = -traceMotion



#%% Plotting

# Plot the traces
functions_analyze.plot_traces(traceNeural,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion)

# Plot auto-correlations
functions_analyze.plot_autocorrelations(traceNeural,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion)

# Plot cross-correlations
functions_analyze.plot_crosscorrelations(traceNeural,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion)










