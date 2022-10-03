# -*- coding: utf-8 -*-
"""
@author: Olivia Gozel

Combining of all datasets for each recording session, and then post-processing
of fluorescence data
(1) ROIs should be at least 5um away from each others, and if a pair
    is too close, only the largest one is kept
(2) Classify ROIs are "visually evoked" or "spontaneously active" by running 
    an anova over the mean df/f0 of all blank frames versus the mean df/f0 of
    all grating frames.
(3) Determine which of the "visually evoked" ROIs are orientation-selective 
    (OS) by one-way anova (ROIs which are not responsive to the grating stimuli
    cannot be OS)
(4) Determine the orientation preference of OS ROIs (not a good tuning since 
    there are only four orientations)
    
This code can be used to postprocess different datasets ('dataType'):
- 'L4_cytosolic': simultaneous L4 cytosolic data with pupil and face motion
- 'ThalamicAxons_L23': simultaneous L2/3 data with thalamic bouton inputs (and behavior)
- 'L4_LGN_targeted_axons'
                    

"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
from os import path

os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_postprocess



#%% Parameters of the data to preprocess

### TO CHOOSE ###
dataType = 'L23_thalamicBoutons' # 'L4_cytosolic' or 'L23_thalamicBoutons'
# 'L4_LGN_targeted_axons'

filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')

### TO CHOOSE ###
idxDataset = 0 # NB: L2/3 dataset4 has weird avg fluo per ori
boolBoutons = True
# bool_discardFluoPlaneSide = False # By default: False; if we need to discard ROIs on the border of the field of view


dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
nBlankFrames = dataSpecs.iloc[idxDataset]['nBlankFrames']
nStimFrames = dataSpecs.iloc[idxDataset]['nStimFrames']
dataFramesPerTrial = nBlankFrames + nStimFrames

### To CHOOSE ###
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor




#%% Load fluorescence data (V1) and classify ROIs

print("Loading fluorescence data (V1)...")
dff0,order,positionROI_3d,nTrialsPerSession = functions_postprocess.loadData(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub,nBlankFrames,nStimFrames)

# Parameters for this dataset
nROI_init,nTrials,fluoPlaneWidth,fluoPlaneHeight = functions_postprocess.determine_params(dff0,positionROI_3d,nBlankFrames,nStimFrames)

print("Classifying neurons...")
charROI,charTrials,fluo,data,positionROI,distROI,idxKept = functions_postprocess.selectROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,order,dff0,positionROI_3d,nBlankFrames,nStimFrames)


#%% Take care of the behavioral data if present

# Pupil data
#charTrials = functions_postprocess.preprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
charPupil = functions_postprocess.postprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path)
if charPupil is not None:
    # Plot the average pupil area per orientation
    data = pd.concat([charTrials,charPupil['pupilArea']],axis=1)
    functions_postprocess.plot_avgPupilPerOri(data,dataFramesPerTrial)
    
# Motion data
#charTrials,motAvg,uMotMask = functions_postprocess.preprocess_motion(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
motSVD = functions_postprocess.postprocess_motion(dataType,dataDate,dataMouse,dataDepth,path)
    

# #%% If needed, artificially discard ROIs which are on the border of the field of view

# if bool_discardFluoPlaneSide:
#     print("Discarding ROIs which are on the border of the field of view!")
#     thresh = int(0.98*fluoPlaneWidth)
#     tmpIdx = np.array(charROI['xCM']<thresh) # boolean vector
#     #zidx = np.array(np.where(tmpIdx==True))[0] # array with indices
#     #fluo = fluo[zidx]
#     fluo = np.array(fluo)[:,tmpIdx]
#     fluo = pd.DataFrame(fluo)
#     #charROI = charROI[tmpIdx]
#     charROI = charROI.iloc[tmpIdx].reset_index()
#     charROI = charROI.drop(columns=['index'])
#     #positionROI = positionROI[tmpIdx]
#     positionROI = positionROI.iloc[tmpIdx].reset_index()
#     positionROI = positionROI.drop(columns=['index'])
#     distROI = distROI[tmpIdx,:][:,tmpIdx]
#     functions_preprocess.plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)
#     data = pd.concat([charTrials, fluo],axis=1)
#     functions_preprocess.plot_avgFluoPerOri(data,dataFramesPerTrial)

#%% Save data

print("Saving all the data...")

savefilepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
    dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'

fluo.to_hdf(savefilepath,key='fluo')
charROI.to_hdf(savefilepath,key='charROI')
charTrials.to_hdf(savefilepath,key='charTrials')
positionROI.to_hdf(savefilepath,key='positionROI')
distROI = pd.DataFrame(distROI)
distROI.to_hdf(savefilepath,key='distROI')

fluoPlaneWidthHeight = np.array([fluoPlaneWidth,fluoPlaneHeight])
fluoPlaneWidthHeight = pd.DataFrame(fluoPlaneWidthHeight)
fluoPlaneWidthHeight.to_hdf(savefilepath,key='fluoPlaneWidthHeight')

if charPupil is not None:
    charPupil.to_hdf(savefilepath,key='charPupil')

if motSVD is not None:
    motSVD.to_hdf(savefilepath,key='motSVD')

# if motAvg is not None:
#     motAvg = pd.DataFrame(motAvg)
#     motAvg.to_hdf(savefilepath,key='motAvg')
    
# if uMotMask is not None:
#     nSVD = uMotMask.shape[2]
#     xtmp = uMotMask.shape[0]
#     ytmp = uMotMask.shape[1]
#     uMotMask = pd.DataFrame(uMotMask.reshape(xtmp*ytmp,nSVD))
#     uMotMask.to_hdf(savefilepath,key='uMotMask')
    

# NB: To read the dataframes within the saved file
# example:  data = pd.read_hdf(filepath,'data')


# Check linear sridge regression score
import sklearn.linear_model as lm
X = np.array(motSVD)
y = np.array(fluo[0])
model = lm.LinearRegression() # lm.Ridge(alpha=0.5)
idxNan = np.where(np.isnan(y))[0]
y = np.delete(y,idxNan)
X = np.delete(X,idxNan,axis=0)
model.fit(X, y)
betaParams = model.coef_
model.score(X, y)





#%% Load thalamic bouton data and select good boutons for further analyses

if boolBoutons:
        
    boutonNeuropilSub = globalParams.neuropilSub[0] # no neuropil subtraction for the thalamic boutons
    
    print("Loading thalamic boutons fluorescence data...")
    nTrialsPerSession,dff0,order,baseline_raw,positionROI_3d = functions_postprocess.loadBoutonData(dataType,dataSessions,dataDate,dataMouse,dataDepth,boutonNeuropilSub,nBlankFrames,nStimFrames)
    
    # Parameters for this dataset
    nROI_init,nTrials,fluoPlaneWidth,fluoPlaneHeight = functions_postprocess.determine_params(dff0,positionROI_3d,nBlankFrames,nStimFrames)
    
    print("Selecting thalamic boutons...")
    charROI,charTrials,fluo,fluo_array,data,positionROI,distROI = functions_postprocess.selectBoutonROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,order,dff0,positionROI_3d,nBlankFrames,nStimFrames)
    
    print("Taking care of the behavioral data if present...")
    # Pupil data
    #charTrials = functions_postprocess.preprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
    charPupil = functions_postprocess.postprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path)
    if charPupil is not None:
        # Plot the average pupil area per orientation
        data = pd.concat([charTrials,charPupil['pupilArea']],axis=1)
        functions_postprocess.plot_avgPupilPerOri(data,dataFramesPerTrial)
    
    # Motion data
    #charTrials,motAvg,uMotMask = functions_postprocess.preprocess_motion(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
    motSVD = functions_postprocess.postprocess_motion(dataType,dataDate,dataMouse,dataDepth,path)
        
    print("Saving all the data...")
    savefilepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + boutonNeuropilSub + '.hdf'
    
    fluo.to_hdf(savefilepath,key='fluo')
    charROI.to_hdf(savefilepath,key='charROI')
    charTrials.to_hdf(savefilepath,key='charTrials')
    positionROI.to_hdf(savefilepath,key='positionROI')
    distROI = pd.DataFrame(distROI)
    distROI.to_hdf(savefilepath,key='distROI')
    fluoPlaneWidthHeight = np.array([fluoPlaneWidth,fluoPlaneHeight])
    fluoPlaneWidthHeight = pd.DataFrame(fluoPlaneWidthHeight)
    fluoPlaneWidthHeight.to_hdf(savefilepath,key='fluoPlaneWidthHeight')
    
    if charPupil is not None:
        charPupil.to_hdf(savefilepath,key='charPupil')

    if motSVD is not None:
        motSVD.to_hdf(savefilepath,key='motSVD')
    
    # if motAvg is not None:
    #     motAvg = pd.DataFrame(motAvg)
    #     motAvg.to_hdf(savefilepath,key='motAvg')
        
    # if uMotMask is not None:
    #     nSVD = uMotMask.shape[2]
    #     xtmp = uMotMask.shape[0]
    #     ytmp = uMotMask.shape[1]
    #     uMotMask = pd.DataFrame(uMotMask.reshape(xtmp*ytmp,nSVD))
    #     uMotMask.to_hdf(savefilepath,key='uMotMask')



# z = motSVD[:,0]
# z2 = np.reshape(z,(2430,5),order='F')

