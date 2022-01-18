# -*- coding: utf-8 -*-
"""
@author: Olivia Gozel

Pre-processing of fluorescence data
(1) ROIs should be active enough ('baseline_raw' of the first session 
    should be higher than the 15th-percentile)
(2) ROIs should be responsive enough to grating stimuli (max over all
    orientations of the average value over all stimulus frames and
    trials should be higher than 10%)
(3) ROIs should be at least 10um away from each others, and if a pair
    is too close, only the largest one is kept
(3bis) ROIs should be big enough (in terms of number of pixels): for now threshold is hard-coded at 1000 pixels
(4) Set a lower bound of 0% to all df/f0 values
(5) Determine if ROIs are orientation-selective (OS) by one-way anova
    and to which orientation they are the most responsive
    
This code can be used to preprocess different datasets ('dataType'):
- 'L4_cytosolic': simultaneous L4 cytosolic data with pupil and face motion
- 'ThalamicAxons_L23': simultaneous L2/3 data with thalamic bouton inputs (and behavior)
                    !! The threshold for Max_dff0 cannot be the same !!
                    
NB: It would be good to discard V1 ROIs which are too small => set a threshold 
on the size of the ROIs. TODO

"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
from os import path

os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_preprocess

dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data\\'


#%% Parameters of the data to preprocess

filepath = dataDir + 'L4_cytosolic_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,'L4_cytosolic_dataSpecs')

dataType = 'L4_cytosolic'
idxDataset = 0
bool_discardFluoPlaneSide = False # By default: False; if we need to discard ROIs on the border of the field of view


dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor




#%% Load fluorescence data (V1): recover all the sessions for one dataset

print("Loading fluorescence data (V1)...")
dff0,order,baseline_raw,positionROI_3d,nTrialsPerSession = functions_preprocess.loadData(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub)

# Parameters for this dataset
nROI_init,nTrials,fluoPlaneWidth,fluoPlaneHeight = functions_preprocess.determine_params(dff0,dataType,positionROI_3d)



#%% Selection of ROIs for further analyses

print("Selecting neurons...")
#charROI,charTrials,fluo,fluo_array,data,percBaselineRaw,positionROI,positionROI_3d,distROI,idxKept = functions_preprocess.selectROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,baseline_raw,order,dff0,positionROI_3d)
charROI,charTrials,fluo,fluo_array,data,percBaselineRaw,positionROI,distROI,idxKept = functions_preprocess.selectROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,baseline_raw,order,dff0,positionROI_3d)

#%% Plotting after getting rid of bad ROIs

# # Plot all kept ROIs at the end of preprocessing
# plt.figure()
# plt.imshow(np.transpose(np.sum(positionROI_3d,axis=0)))
# plt.title('ROIs kept after preprocessing')

# # Plot all kept ROIs with their selectivity
# functions_preprocess.plot_ROIwithSelectivity(fluoPlaneHeight,fluoPlaneWidth,charROI,positionROI_3d)

# # Plot the average fluorescence for each trial orientation
# functions_preprocess.plot_avgFluoPerOri(dataType,data)



#%% Take care of the behavioral data if present

# Pupil data
charTrials = functions_preprocess.preprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
    
# Motion data
charTrials,motAvg,uMotMask = functions_preprocess.preprocess_motion(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
    

#%% If needed, artificially discard ROIs which are on the border of the field of view

if bool_discardFluoPlaneSide:
    print("Discarding ROIs which are on the border of the field of view!")
    thresh = int(0.98*fluoPlaneWidth)
    tmpIdx = np.array(charROI['xCM']<thresh) # boolean vector
    zidx = np.array(np.where(tmpIdx==True))[0] # array with indices
    fluo = fluo[zidx]
    charROI = charROI[tmpIdx]
    positionROI = positionROI[tmpIdx]
    distROI = distROI[tmpIdx,:][:,tmpIdx]
    functions_preprocess.plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)


#%% Save data

print("Saving all the data...")

savefilepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
    dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist2d5um.hdf'

fluo.to_hdf(savefilepath,key='fluo')
charROI.to_hdf(savefilepath,key='charROI')
charTrials.to_hdf(savefilepath,key='charTrials')
positionROI.to_hdf(savefilepath,key='positionROI')
distROI = pd.DataFrame(distROI)
distROI.to_hdf(savefilepath,key='distROI')


fluoPlaneWidthHeight = np.array([fluoPlaneWidth,fluoPlaneHeight])
fluoPlaneWidthHeight = pd.DataFrame(fluoPlaneWidthHeight)
fluoPlaneWidthHeight.to_hdf(savefilepath,key='fluoPlaneWidthHeight')

if motAvg is not None:
    motAvg = pd.DataFrame(motAvg)
    motAvg.to_hdf(savefilepath,key='motAvg')
    
if uMotMask is not None:
    nSVD = uMotMask.shape[2]
    xtmp = uMotMask.shape[0]
    ytmp = uMotMask.shape[1]
    uMotMask = pd.DataFrame(uMotMask.reshape(xtmp*ytmp,nSVD))
    uMotMask.to_hdf(savefilepath,key='uMotMask')
    

# NB: To read the dataframes within the saved file
# example:  data = pd.read_hdf(filepath,'data')















