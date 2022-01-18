# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:22:31 2021

@author: Olivia Gozel

Analyze the V1 and boutons data (one piece at a time)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze
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



#pieceNum = 1234


#%% Load data

#filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
#        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_V1andBoutons_piece' + str(pieceNum) + '.hdf'

#filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
#            dataMouse + '_' + dataDepth + '_V1nF0d75_LGNnF0_V1andBoutons_piece' + str(pieceNum) + '.hdf'
filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
            dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'

# V1 data
fluo = pd.read_hdf(filepath,'fluo')
charROI = pd.read_hdf(filepath,'charROI')
charTrials = pd.read_hdf(filepath,'charTrials')
positionROI = pd.read_hdf(filepath,'positionROI')
distROI = pd.read_hdf(filepath,'distROI')
nROI = fluo.shape[1]
#idxKept = np.squeeze(np.asarray(pd.read_hdf(filepath,'idxKept')))

fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()

# # Boutons data
# fluo_boutons = pd.read_hdf(filepath,'fluo_boutons')
# charBoutons = pd.read_hdf(filepath,'charROI_boutons')
# charTrials_boutons = pd.read_hdf(filepath,'charTrials_boutons')
# positionBoutons = pd.read_hdf(filepath,'positionROI_boutons')
# distBoutons = pd.read_hdf(filepath,'distROI_boutons')
# nBoutons = fluo_boutons.shape[1]
# #idxKept_boutons = np.squeeze(np.asarray(pd.read_hdf(filepath,'idxKept_boutons')))


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




#%% Plot ROIs with their selectivity


# V1
functions_preprocess.plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)

# Thalamic boutons
functions_preprocess.plot_ROIwithSelectivity(charBoutons,positionBoutons,fluoPlaneWidth,fluoPlaneHeight)

# L2/3 ROIs and thalamic boutons
functions_preprocess.plot_ROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight)


#%% Determine what is the percentage of OS V1 ROI and the percentage of OS thalamic boutons

percROI_OS = 100*(charROI[charROI['OS']==True].shape[0] / charROI.shape[0])
percBouton_OS = 100*(charBoutons[charBoutons['OS']==True].shape[0] / charBoutons.shape[0])
print(str(np.around(percROI_OS))+'% ROIs are OS, and '+str(np.around(percBouton_OS))+'% thalamic boutons are OS.')



#%% Plot the pairwise correlation as a function of pairwise distance

## For L2/3 ROIs
idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
thisDistROI = np.array(distROI)[idxOS,:][:,idxOS]

# Evoked frames (OS neurons only)
idxStim = np.squeeze(np.array(np.where(charTrials['FrameType']=='Stimulus')))
thisFluo = np.array(fluo)[idxStim,:][:,idxOS]

binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='L2/3 - Evoked - OS only')


# Spontaneous frames (OS neurons only)
idxBlank = np.squeeze(np.array(np.where(charTrials['FrameType']=='Blank')))
thisFluo = np.array(fluo)[idxBlank,:][:,idxOS]

binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='L2/3 - Spontaneous - OS only')


## For thalamic bouton inputs
idxOS = np.squeeze(np.array(np.where(np.array(charBoutons['OS']))))
thisDistROI = np.array(distBoutons)[idxOS,:][:,idxOS]

# Evoked frames (OS neurons only)
idxStim = np.squeeze(np.array(np.where(charTrials['FrameType']=='Stimulus')))
thisFluo = np.array(fluo_boutons)[idxStim,:][:,idxOS]

binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='Thalamic boutons - Evoked - OS only')


# Spontaneous frames (OS neurons only)
idxBlank = np.squeeze(np.array(np.where(charTrials['FrameType']=='Blank')))
thisFluo = np.array(fluo_boutons)[idxBlank,:][:,idxOS]

binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='Thalamic boutons - Spontaneous - OS only')


#%% Compute the neural PCs

# Select the ROIs we are interested in
idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
#thisFluo = np.array(fluo)
thisFluo = np.array(fluo)[:,idxOS]

# All frames, only OS neurons
neuralProj_L23, percVarExpl_L23 = functions_analyze.compute_neuralPCs(thisFluo,bool_plot=1,title='L2/3 - All frames - OS only')


# Select the ROIs we are interested in
idxOS = np.squeeze(np.array(np.where(np.array(charBoutons['OS']))))
#thisFluo = np.array(fluo)
thisFluo = np.array(fluo_boutons)[:,idxOS]

# All frames, only OS neurons
neuralProj_boutons, percVarExpl_boutons = functions_analyze.compute_neuralPCs(thisFluo,bool_plot=1,title='Thalamic boutons - All frames - OS only')



#%% Define the traces of interest

traceNeural_L23 = neuralProj_L23[:,0]
traceNeural_boutons = neuralProj_boutons[:,0]
if bool_pupil:
    tracePupil = np.array(charTrials['pupilArea']).astype(float)
else:
     tracePupil = None   
if bool_motion:
    traceMotion = np.array(charTrials['motSVD1'])
    # Flip trace if average is negative
    if np.mean(traceMotion) < 0:
        traceMotion = -traceMotion
else:
    traceMotion = None


#%% Plot the traces

# L2/3
functions_analyze.plot_traces(traceNeural_L23,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion)

# Thalamic boutons
functions_analyze.plot_traces(traceNeural_boutons,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion)



#%% Cross-correlation between neural activity and behavior

## For L2/3 ROIs
functions_analyze.plot_crosscorrelations(traceNeural_L23,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion)

## For thalamic boutons
functions_analyze.plot_crosscorrelations(traceNeural_boutons,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion)




# #%% Assign boutons to V1 ROIs -- not so useful...

# # Plot V1 ROIs with the boutons
# functions_preprocess.plot_ROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight)

# # Recover the indices of the overlapping boutons (and the V1 ROI they connect to)
# idxAssignedROI,idxOverlappingBoutons = functions_preprocess.getOverlappingBoutonROI(positionROI,positionBoutons)
    

# # Plot V1 ROIs with the overlapping boutons
# functions_preprocess.plot_ROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight,idxOverlappingBoutons)

# # Plot each ROI with its associated thalamic boutons separately (sanity check)
# functions_preprocess.plot_eachROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight,idxOverlappingBoutons,idxAssignedROI)




# #%% Compare orientation selectivity of ROIs and their assigned thalamic boutons -- does not make sense....

# # Orientation selectivity of ROIs
# theseROI = np.unique(idxAssignedROI)
# theseROI_prefOri = np.asarray(charROI.loc[theseROI,'PrefOri'])

# # Orientation selectivity of assigned thalamic boutons
# theseBoutons_prefOri = np.asarray(charBoutons.loc[idxOverlappingBoutons,'PrefOri'])

# # Compare preferred orientation of each ROI and its assigned boutons
# goodPairs = []
# badPairs = []
# for r in range(0,theseROI.size):
#     thisROI_prefOri = theseROI_prefOri[r]
#     if charROI.loc[theseROI[r],'OS']:
#         thisDiff = np.abs(thisROI_prefOri - theseBoutons_prefOri)
#         tmp = np.asarray(np.where(idxAssignedROI==theseROI[r])).flatten()
#         goodPairs = np.append(goodPairs,thisDiff[tmp])
#         tmp = np.asarray(np.where(idxAssignedROI!=theseROI[r])).flatten()
#         badPairs = np.append(badPairs,thisDiff[tmp])

# # Plot histograms
# plt.figure()
# plt.hist(goodPairs,align='left')
# plt.xticks(np.unique(goodPairs))
# plt.title('PrefOri ROI vs prefOri assigned boutons')

# plt.figure()
# plt.hist(badPairs,align='left')
# plt.xticks(np.unique(badPairs))
# plt.title('PrefOri ROI vs prefOri non-assigned boutons')

# # Percentage of OS ROI and OS boutons
# percROI_OS = 100*(charROI[charROI['OS']==True].shape[0] / charROI.shape[0])
# percBouton_OS = 100*(charBoutons[charBoutons['OS']==True].shape[0] / charBoutons.shape[0])
# print(str(np.around(percROI_OS))+'% ROIs are OS, and '+str(np.around(percBouton_OS))+'% thalamic boutons are OS.')



# #%% Cross-correlation between V1 ROI activity and the activity of thalamic boutons

# # Assigned thalamic boutons
# functions_analyze.plot_crosscorrelations_ROIvsBoutons(fluo,fluo_boutons,idxAssignedROI,idxOverlappingBoutons)


# # Uncoupled thalamic boutons




# #%% Compare variability of ROIs and variability of thalamic boutons

# fluo_movingStd = fluo_boutons.rolling(5,axis=0,center=True).std()
# fluo_movingMean = fluo_boutons.rolling(5,axis=0,center=True).mean()
# stdOverMean_perROI = fluo_movingStd / fluo_movingMean

# # Find out which boutons are extremely variable
# threshold_stdOverMean = 1
# tmpMean = stdOverMean_perROI.mean()
# idxLargeStdOverMean = np.squeeze(np.asarray(np.where(tmpMean>threshold_stdOverMean)))
# idxOkStdOverMean = np.squeeze(np.asarray(np.where(tmpMean<=threshold_stdOverMean)))
# #tmpMean[idxLargeStdOverMean]

# plt.figure()
# plt.plot(stdOverMean_perROI)
# plt.xlabel('Frame')
# plt.ylabel('std/mean')
# plt.title('All 5 sessions, all kept boutons')

# plt.figure()
# plt.plot(stdOverMean_perROI[idxLargeStdOverMean])
# plt.xlabel('Frame')
# plt.ylabel('std/mean')
# plt.title('All 5 sessions, 12 highly variable boutons')

# plt.figure()
# plt.plot(stdOverMean_perROI[idxOkStdOverMean])
# plt.xlabel('Frame')
# plt.ylabel('std/mean')
# plt.title('All 5 sessions, 466 not so variable boutons')

# plt.figure()
# plt.plot(stdOverMean_perROI.loc[0:6000])
# plt.xlabel('Frame')
# plt.ylabel('std/mean')
# plt.title('First session')

# tmp = stdOverMean_perROI.loc[0:6000]
# boutonIDX = 7
# tmp1 = tmp[boutonIDX]
# plt.figure()
# plt.plot(tmp1)
# plt.xlabel('Frame')
# plt.ylabel('stdOverMean')
# plt.title('First session, bouton '+str(boutonIDX+1))


# tmp = stdOverMean_perROI.loc[0:6000]
# tmpMean = tmp.mean()
# tmpSEM = tmp.sem(ddof=0)
# plt.figure()
# plt.errorbar(np.arange(1,tmp.shape[1]+1),tmpMean,tmpSEM)
# plt.xlabel('Bouton index')
# plt.ylabel('Average stdOverMean over session 1 +/- SEM')

# plt.figure()
# plt.plot(tmpMean)
# plt.xlabel('Bouton index')
# plt.ylabel('Mean stdOverMean')
# plt.title('Session 1')



# def plotFanoFactor(fluo,nFramesPerTrial,idx=None):
    
#     if idx is not None:
#         fluo = fluo[idx]
    
#     tmpTime = np.arange(nFramesPerTrial)+1
    
#     # Compute Fano Factor: FF = sigma^2 / mu (computed on a time window)
#     fluo_movingVar = fluo.rolling(5,axis=0,center=True).var()
#     fluo_movingMean = fluo.rolling(5,axis=0,center=True).mean()
    
#     ff_perROI = fluo_movingVar / fluo_movingMean
    
#     # mean over all ROIs
#     #meanff = np.reshape(np.asarray(ff_perROI.mean(axis=1)),(30,-1))
#     #semff = np.reshape(np.asarray(ff_perROI.var(axis=1)/np.sqrt(ff_perROI.shape[1])),(30,-1))
    
#     # Reshape
#     tmpFFperROI = np.reshape(np.asarray(ff_perROI),(nFramesPerTrial,-1))
    
#     # mean over all trials
#     meanff = np.nanmean(tmpFFperROI,axis=1)
#     semff = np.nanvar(tmpFFperROI,axis=1)/np.sqrt(tmpFFperROI.shape[1])
    
#     # Plot average (over all trials) Fano Factor
#     plt.figure()
#     plt.plot(tmpTime,meanff)
#     plt.fill_between(tmpTime,meanff-semff,meanff+semff)
#     plt.xlabel('Frame')
#     plt.ylabel('Fano Factor')


# plotFanoFactor(fluo,30,idxAssignedROI[0])




