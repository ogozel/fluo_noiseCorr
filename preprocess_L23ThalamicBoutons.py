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
                    

"""

import numpy as np
import pandas as pd
#import scipy.io
# from scipy.ndimage import center_of_mass
# import scipy.stats as stats
# from sklearn.metrics.pairwise import euclidean_distances
#import random as rnd
import matplotlib.pyplot as plt
#import matplotlib as mpl
#from matplotlib import colors
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import os.path
import os
from os import path

os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_preprocess

dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data\\'


#%% Parameters of the data to preprocess

filepath = dataDir + 'L23_thalamicBoutons_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,'L23_thalamicBoutons_dataSpecs')

dataType = 'ThalamicAxons_L23'
idxDataset = 3


dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor for the L2/3 cytosolic data
dataNeuropilSub_boutons = globalParams.neuropilSub[0] # choose a neuropil factor for the thalamic bouton data






#%% Load fluorescence data (thalamic bouton inputs): recover all the sessions for one dataset, for each piece separately

    
# Preprocess all the sessions for each of the 4 pieces separately
for pieceNum in range(1,5): # data is in 4 pieces
    
    print('Preprocessing piece ',str(pieceNum),' of 4')
    
    print("Loading fluorescence data (L2/3 of V1)...")
    dff0,order,baseline_raw,positionROI_3d,nTrialsPerSession = functions_preprocess.loadData(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub,thisPieceNum=pieceNum) # load only 1 piece at a time
    
    # Parameters for the V1 fluorescence data
    nROI_init,nTrials,fluoPlaneWidth,fluoPlaneHeight = functions_preprocess.determine_params(dff0,dataType,positionROI_3d)
    
    print("Selecting L2/3 neurons of V1...")
    charROI,charTrials,fluo,fluo_array,data,percBaselineRaw,positionROI,distROI,idxKept = functions_preprocess.selectROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,baseline_raw,order,dff0,positionROI_3d)

    
    print('Loading fluorescence data (thalamic boutons)...')
    nTrialsPerSession_boutons,dff0_boutons,order_boutons,baseline_raw_boutons,positionBoutons_3d = functions_preprocess.loadBoutonData(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub_boutons,pieceNum)          
    
    # Parameters for the thalamic boutons fluorescence data
    nBoutons_init,_,_,_ = functions_preprocess.determine_params(dff0_boutons,dataType,positionBoutons_3d)
    
    
    print("Selecting thalamic boutons...")
    charBoutons,charTrials_boutons,fluo_boutons,fluo_array_boutons,data_boutons,percBaselineRaw_boutons,positionBoutons,positionBoutons_3d,distROI_boutons,idxKept_boutons = functions_preprocess.selectBoutonROIs(dataType,pixelSize,dataSessions,nTrialsPerSession_boutons,baseline_raw_boutons,order_boutons,dff0_boutons,positionBoutons_3d)
    
    # Plot V1 ROIs with the boutons
    functions_preprocess.plot_ROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight)
    
    
    print("Taking care of the behavioral data...")
    # Pupil data
    charTrials = functions_preprocess.preprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
    # Motion data
    charTrials,motAvg,uMotMask = functions_preprocess.preprocess_motion(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
   
    
    # Save data together
    print("Saving the data...")
    #savefilepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        #dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_V1andBoutons_piece' + str(pieceNum) + '.hdf'
    savefilepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_V1nF0d75_LGNnF0_V1andBoutons_piece' + str(pieceNum) + '.hdf'
    
    # V1 data
    fluo.to_hdf(savefilepath,key='fluo')
    charROI.to_hdf(savefilepath,key='charROI')
    charTrials.to_hdf(savefilepath,key='charTrials')
    positionROI.to_hdf(savefilepath,key='positionROI')
    distROI = pd.DataFrame(distROI)
    distROI.to_hdf(savefilepath,key='distROI')
    idxKept = pd.DataFrame(idxKept)
    idxKept.to_hdf(savefilepath,key='idxKept')
    
    fluoPlaneWidthHeight = np.array([fluoPlaneWidth,fluoPlaneHeight])
    fluoPlaneWidthHeight = pd.DataFrame(fluoPlaneWidthHeight)
    fluoPlaneWidthHeight.to_hdf(savefilepath,key='fluoPlaneWidthHeight')
    
    # Thalamic boutons data
    fluo_boutons.to_hdf(savefilepath,key='fluo_boutons')
    charBoutons.to_hdf(savefilepath,key='charROI_boutons')
    charTrials_boutons.to_hdf(savefilepath,key='charTrials_boutons')
    positionBoutons.to_hdf(savefilepath,key='positionROI_boutons')
    distROI_boutons = pd.DataFrame(distROI_boutons)
    distROI_boutons.to_hdf(savefilepath,key='distROI_boutons')
    idxKept_boutons = pd.DataFrame(idxKept_boutons)
    idxKept_boutons.to_hdf(savefilepath,key='idxKept_boutons')

    # Motion data
    if motAvg is not None:
        motAvg = pd.DataFrame(motAvg)
        motAvg.to_hdf(savefilepath,key='motAvg')
    
    if uMotMask is not None:
        nSVD = uMotMask.shape[2]
        xtmp = uMotMask.shape[0]
        ytmp = uMotMask.shape[1]
        uMotMask = pd.DataFrame(uMotMask.reshape(xtmp*ytmp,nSVD))
        uMotMask.to_hdf(savefilepath,key='uMotMask')



#%% Combine the 4 pieces together

from sklearn.metrics.pairwise import euclidean_distances

for pieceNum in range(1,5): # data is in 4 pieces
    
    # Load data
    #filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        #dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_V1andBoutons_piece' + str(pieceNum) + '.hdf'
    filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
            dataMouse + '_' + dataDepth + '_V1nF0d75_LGNnF0_V1andBoutons_piece' + str(pieceNum) + '.hdf'
        
    fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
    fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
    fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()
        
    # V1 data
    this_fluo = pd.read_hdf(filepath,'fluo')
    this_charROI = pd.read_hdf(filepath,'charROI')
    #charTrials = pd.read_hdf(filepath,'charTrials')
    positionROI = np.array(pd.read_hdf(filepath,'positionROI'))
    positionROI_3d = positionROI.reshape((positionROI.shape[0],fluoPlaneWidth,fluoPlaneHeight))
    #distROI = pd.read_hdf(filepath,'distROI')
    #nROI = fluo.shape[1]
    #idxKept = np.squeeze(np.asarray(pd.read_hdf(filepath,'idxKept')))
    
    # Boutons data
    this_fluo_boutons = pd.read_hdf(filepath,'fluo_boutons')
    this_charBoutons = pd.read_hdf(filepath,'charROI_boutons')
    #charTrials_boutons = pd.read_hdf(filepath,'charTrials_boutons')
    positionBoutons = np.array(pd.read_hdf(filepath,'positionROI_boutons'))
    positionBoutons_3d = positionBoutons.reshape((positionBoutons.shape[0],fluoPlaneWidth,fluoPlaneHeight))
    #distBoutons = pd.read_hdf(filepath,'distROI_boutons')
    #nBoutons = fluo_boutons.shape[1]
    #idxKept_boutons = np.squeeze(np.asarray(pd.read_hdf(filepath,'idxKept_boutons')))
    
    if pieceNum==1:
        charROI = this_charROI
        charBoutons = this_charBoutons
        fluo = this_fluo
        fluo_boutons = this_fluo_boutons
        this_positionROI3d = np.pad(positionROI_3d,((0,0),(0,1024),(0,1024))) # piece1
        this_positionBoutons3d = np.pad(positionBoutons_3d,((0,0),(0,1024),(0,1024))) # piece1
    else:
        if pieceNum==2:
            tmp = np.pad(positionROI_3d,((0,0),(1024,0),(0,1024))) # piece2
            tmp2 = np.pad(positionBoutons_3d,((0,0),(1024,0),(0,1024))) # piece2
            this_charROI['xCM'] = this_charROI['xCM']+1024
            this_charBoutons['xCM'] = this_charBoutons['xCM']+1024
        if pieceNum==3:
            tmp = np.pad(positionROI_3d,((0,0),(0,1024),(1024,0))) # piece3
            tmp2 = np.pad(positionBoutons_3d,((0,0),(0,1024),(1024,0))) # piece3
            this_charROI['yCM'] = this_charROI['yCM']+1024
            this_charBoutons['yCM'] = this_charBoutons['yCM']+1024
        if pieceNum==4:
            tmp = np.pad(positionROI_3d,((0,0),(1024,0),(1024,0))) # piece4
            tmp2 = np.pad(positionBoutons_3d,((0,0),(1024,0),(1024,0))) # piece4
            this_charROI['xCM'] = this_charROI['xCM']+1024
            this_charROI['yCM'] = this_charROI['yCM']+1024
            this_charBoutons['xCM'] = this_charBoutons['xCM']+1024
            this_charBoutons['yCM'] = this_charBoutons['yCM']+1024
        this_positionROI3d = np.concatenate((this_positionROI3d,tmp),axis=0)
        this_positionBoutons3d = np.concatenate((this_positionBoutons3d,tmp2),axis=0)
        charROI = pd.concat([charROI, this_charROI])
        charBoutons = pd.concat([charBoutons, this_charBoutons])
        fluo = pd.concat([fluo,this_fluo],axis=1,ignore_index=True)
        fluo_boutons = pd.concat([fluo_boutons,this_fluo_boutons],axis=1,ignore_index=True)
    
    if pieceNum==1: # same for all pieces
        
        charTrials = pd.read_hdf(filepath,'charTrials')
        charTrials_boutons = pd.read_hdf(filepath,'charTrials_boutons')
        
        if 'pupilArea' in charTrials:
            bool_pupil = True
        else:
            bool_pupil = False
        
        store = pd.HDFStore(filepath,mode='r')
        if 'motAvg' in store:
            bool_motion = True
            motAvg = pd.read_hdf(filepath,'motAvg')
            uMotMask = pd.read_hdf(filepath,'uMotMask')
        else:
            bool_motion = False

nROI = len(charROI)
positionROI = pd.DataFrame(this_positionROI3d.reshape(nROI,2048*2048))
charROI = charROI.reset_index()
for n in range(nROI):
    if n==0:
        cmROI = np.array([[charROI.iloc[n]['xCM'],charROI.iloc[n]['yCM']]])
    else:
        cmROI = np.append(cmROI,np.array([[charROI.iloc[n]['xCM'],charROI.iloc[n]['yCM']]]),axis=0)

# Compute pairwise distance between all ROIs
distROI = pixelSize*euclidean_distances(cmROI)

nBoutons = len(charBoutons)
positionBoutons = pd.DataFrame(this_positionBoutons3d.reshape(nBoutons,2048*2048))
charBoutons = charBoutons.reset_index()
for n in range(nBoutons):
    if n==0:
        cmBoutons = np.array([[charBoutons.iloc[n]['xCM'],charBoutons.iloc[n]['yCM']]])
    else:
        cmBoutons = np.append(cmBoutons,np.array([[charBoutons.iloc[n]['xCM'],charBoutons.iloc[n]['yCM']]]),axis=0)

# Compute pairwise distance between all ROIs
distROI_boutons = pixelSize*euclidean_distances(cmBoutons)


# Save all pieces data together
print("Saving the data...")
#savefilepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
#    dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_V1andBoutons_piece' + str(1234) + '.hdf'
savefilepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
    dataMouse + '_' + dataDepth + '_V1nF0d75_LGNnF0_V1andBoutons_piece' + str(1234) + '.hdf'

# V1 data
fluo.to_hdf(savefilepath,key='fluo')
charROI.to_hdf(savefilepath,key='charROI')
charTrials.to_hdf(savefilepath,key='charTrials')
positionROI.to_hdf(savefilepath,key='positionROI')
distROI = pd.DataFrame(distROI)
distROI.to_hdf(savefilepath,key='distROI')
#idxKept = pd.DataFrame(idxKept)
#idxKept.to_hdf(savefilepath,key='idxKept')

fluoPlaneWidthHeight = np.array([2048,2048])
fluoPlaneWidthHeight = pd.DataFrame(fluoPlaneWidthHeight)
fluoPlaneWidthHeight.to_hdf(savefilepath,key='fluoPlaneWidthHeight')

# Thalamic boutons data
fluo_boutons.to_hdf(savefilepath,key='fluo_boutons')
charBoutons.to_hdf(savefilepath,key='charROI_boutons')
charTrials_boutons.to_hdf(savefilepath,key='charTrials_boutons')
positionBoutons.to_hdf(savefilepath,key='positionROI_boutons')
distROI_boutons = pd.DataFrame(distROI_boutons)
distROI_boutons.to_hdf(savefilepath,key='distROI_boutons')
#idxKept_boutons = pd.DataFrame(idxKept_boutons)
#idxKept_boutons.to_hdf(savefilepath,key='idxKept_boutons')

# Motion data
if motAvg is not None:
    motAvg = pd.DataFrame(motAvg)
    motAvg.to_hdf(savefilepath,key='motAvg')

if uMotMask is not None:
    # nSVD = uMotMask.shape[2]
    # xtmp = uMotMask.shape[0]
    # ytmp = uMotMask.shape[1]
    # uMotMask = pd.DataFrame(uMotMask.reshape(xtmp*ytmp,nSVD))
    uMotMask.to_hdf(savefilepath,key='uMotMask')











#%% Load fluorescence data (V1): recover all the sessions for one dataset

print("Loading fluorescence data (V1)...")
dff0,order,baseline_raw,positionROI_3d,nTrialsPerSession = functions_preprocess.loadData(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub)

# Parameters for this dataset
nROI_init,nTrials,fluoPlaneWidth,fluoPlaneHeight = functions_preprocess.determine_params(dff0,dataType,positionROI_3d)



#%% Selection of ROIs for further analyses

print("Selecting neurons...")
charROI,charTrials,fluo,fluo_array,data,percBaselineRaw,positionROI,distROI,idxKept = functions_preprocess.selectROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,baseline_raw,order,dff0,positionROI_3d)



#%% Take care of the behavioral data if present

# Pupil data
charTrials = functions_preprocess.preprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
    
# Motion data
charTrials,motAvg,uMotMask = functions_preprocess.preprocess_motion(dataType,dataDate,dataMouse,dataDepth,path,charTrials)
    


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



