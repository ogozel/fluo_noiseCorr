# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:46:09 2021

@author: Olivia Gozel

Analyze the data using home-made functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze
import functions_preprocess




#%% Parameters of the preprocessed data to analyze

### TO CHOOSE ###
dataType = 'L23_thalamicBoutons' # 'L4_cytosolic' or 'L23_thalamicBoutons'

filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')

### TO CHOOSE ###
idxDataset = 0 # L4_cytosolic: 0,1,4,8: nice corr=f(dist); 2,3,5,6,7,9: bad corr=f(dist)

dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
dataFR = dataSpecs.iloc[idxDataset]['FrameRate']

### To CHOOSE ###
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor


#%% Load data

filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist2d5um.hdf'

fluoINIT = pd.read_hdf(filepath,'fluo')
charROI = pd.read_hdf(filepath,'charROI')
charTrials = pd.read_hdf(filepath,'charTrials')
positionROI = pd.read_hdf(filepath,'positionROI')
distROI = pd.read_hdf(filepath,'distROI')
fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()

fluo = fluoINIT

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

functions_preprocess.plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)

percROI_OS = 100*(charROI[charROI['OS']==True].shape[0] / charROI.shape[0])
print(str(np.around(percROI_OS))+'% V1 ROIs are orientation-selective (OS).')


#%% Compute grand average fluorescence traces

# Some ROIs are extremely variable en biais completely the results: we need to discard them
varFluo = np.var(np.array(fluo),axis=0)
idxToKeep = np.where(varFluo < np.median(varFluo,axis=0) + 3*np.std(varFluo,axis=0))[0]
thisFluo = fluo[idxToKeep]
thisCharROI = charROI.iloc[idxToKeep].reset_index()

# All ROIs
functions_analyze.plot_avgFluoPerOri(dataType,charTrials,thisFluo,dataFR,title='All ROIs')

# Only ROIs selective for a given orientation
for o in range(4):
    tmpIdx = np.where((thisCharROI['OS']==True)&(thisCharROI['PrefOri']==globalParams.ori[o]))[0]
    tmpFluo = thisFluo[idxToKeep[tmpIdx]]
    functions_analyze.plot_avgFluoPerOri(dataType,charTrials,tmpFluo,dataFR,title='ROIs selective for '+str(globalParams.ori[o])+'°')
    
# Non-OS ROIs only
idxNonOS = np.where(thisCharROI['OS']==False)[0]
tmpFluo = thisFluo[idxToKeep[idxNonOS]]
functions_analyze.plot_avgFluoPerOri(dataType,charTrials,tmpFluo,dataFR,title='Non-OS ROIs')

# OS ROIs only
idxOS = np.where(thisCharROI['OS']==True)[0]
tmpFluo = thisFluo[idxToKeep[idxOS]]
functions_analyze.plot_avgFluoPerOri(dataType,charTrials,tmpFluo,dataFR,title='OS ROIs')


#%% Compute grand average behavioral trace

functions_analyze.plot_avgBehavioralTrace(dataType,charTrials,dataFR,bool_motion,bool_pupil)



#%% Compute Fano Factor and plot its grand average

# Some ROIs are extremely variable en biais completely the results: we need to discard them
varFluo = np.var(np.array(fluo),axis=0)
idxToKeep = np.where(varFluo < np.median(varFluo,axis=0) + 3*np.std(varFluo,axis=0))[0]
thisFluo = fluo[idxToKeep]
thisCharROI = charROI.iloc[idxToKeep].reset_index()

# All ROIs
fanoFactor, fanoFactorPerOri = functions_analyze.compute_fanoFactor(dataType,charTrials,thisFluo,title='All ROIs')

# Only ROIs selective for a given orientation
for o in range(4):
    tmpIdx = np.where((thisCharROI['OS']==True)&(thisCharROI['PrefOri']==globalParams.ori[o]))[0]
    tmpFluo = thisFluo[idxToKeep[tmpIdx]]
    fanoFactor, fanoFactorPerOri = functions_analyze.compute_fanoFactor(dataType,charTrials,tmpFluo,title='ROIs selective for '+str(globalParams.ori[o])+'°')
    
# Non-OS ROIs only
idxNonOS = np.where(thisCharROI['OS']==False)[0]
tmpFluo = thisFluo[idxToKeep[idxNonOS]]
fanoFactor, fanoFactorPerOri = functions_analyze.compute_fanoFactor(dataType,charTrials,tmpFluo,title='Non-OS ROIs')

# OS ROIs only
idxOS = np.where(thisCharROI['OS']==True)[0]
tmpFluo = thisFluo[idxToKeep[idxOS]]
fanoFactor, fanoFactorPerOri = functions_analyze.compute_fanoFactor(dataType,charTrials,tmpFluo,title='OS ROIs')



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

binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,startBin=2.5,binSize=10,title='Spontaneous - OS only')



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
if 'pupilArea' in charTrials:
    tracePupil = np.array(charTrials['pupilArea']).astype(float)
else:
    tracePupil = None
if 'motSVD1' in charTrials:
    traceMotion = np.array(charTrials['motSVD1'])
else:
    traceMotion = None

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



#%% Within-area full and shared dimensionality

fluo = np.array(fluoINIT)
varROI = np.var(fluoINIT,axis=0)
plt.figure()
plt.hist(varROI)
plt.title('Variability of ROIs')
idxTooVar = np.where(varROI>np.median(varROI,axis=0) + 3*np.std(varROI,axis=0))[0]  # np.where(varROI>1e7)[0]
fluo = np.delete(fluo,idxTooVar,axis=1)
charROI = charROI.iloc[np.setdiff1d(np.arange(charROI.shape[0]),idxTooVar)].reset_index()

# Select the ROIs and frame type we are interested i
idxOS = np.where(np.array(charROI['OS']))[0]
idxNonOS = np.where(np.array(charROI['OS']==False))[0]
idxBlank = np.where(charTrials['FrameType']=='Blank')[0]
idxStimulus = np.where(charTrials['FrameType']=='Stimulus')[0]

thisType='private' # 'full','shared','private'

# OS only, blank frames
thisFluo = fluo[:,idxOS][idxBlank,:]
partRatio, covBOS = functions_analyze.compute_dimensionality(thisFluo,type=thisType)
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[25,50,75,100,125,len(idxOS)],nTimes=50,type='full',boolPlot=True)

# OS only, stimulus frames
thisFluo = fluo[:,idxOS][idxStimulus,:]
partRatio, covSOS = functions_analyze.compute_dimensionality(thisFluo,type=thisType)
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[25,50,75,100,125,len(idxOS)],nTimes=50,type='full',boolPlot=True)

# Non-OS only, blank frames
thisFluo = fluo[:,idxNonOS][idxBlank,:]
partRatio, covBnonOS = functions_analyze.compute_dimensionality(thisFluo,type=thisType)

# Non-OS only, stimulus frames
thisFluo = fluo[:,idxNonOS][idxStimulus,:]
partRatio, covSnonOS = functions_analyze.compute_dimensionality(thisFluo,type=thisType)

thisBoolPlot=False
if thisBoolPlot:
    # Compare diagonal elements of covariance matrices between spontaneous and evoked
    plt.figure()
    plt.hist(np.diag(covBOS),alpha=0.5) # ,bins=np.arange(0,200,25)
    plt.hist(np.diag(covSOS),alpha=0.5)
    plt.legend(['Blank, mu='+str(np.round(np.mean(np.diag(covBOS)),1)),'Stimulus, mu='+str(np.round(np.mean(np.diag(covSOS)),1))])
    plt.xlabel(thisType+' variance')
    plt.title('OS only')
    
    plt.figure()
    plt.hist(np.diag(covBnonOS),alpha=0.5)
    plt.hist(np.diag(covSnonOS),alpha=0.5)
    plt.legend(['Blank, mu='+str(np.round(np.mean(np.diag(covBnonOS)),1)),'Stimulus, mu='+str(np.round(np.mean(np.diag(covSnonOS)),1))])
    plt.xlabel(thisType+' variance')
    plt.title('Non-OS only')
    
    # Compare non-diagonal elements of covariance matrices between spontaneous and evoked
    plt.figure()
    plt.hist( covBOS[np.triu_indices(covBOS.shape[0],k=0)], alpha=0.5)
    plt.hist( covSOS[np.triu_indices(covSOS.shape[0],k=0)], alpha=0.5)
    plt.legend(['Blank, mu='+str(np.round(np.mean(covBOS[np.triu_indices(covBOS.shape[0],k=0)]))),'Stimulus, mu='+str(np.round(np.mean(covSOS[np.triu_indices(covSOS.shape[0],k=0)])))])
    plt.xlabel(thisType+' covariance')
    plt.title('OS only')
    
    plt.figure()
    plt.hist( covBnonOS[np.triu_indices(covBnonOS.shape[0],k=0)], alpha=0.5)
    plt.hist( covSnonOS[np.triu_indices(covSnonOS.shape[0],k=0)], alpha=0.5)
    plt.legend(['Blank, mu='+str(np.round(np.mean(covBnonOS[np.triu_indices(covBnonOS.shape[0],k=0)]))),'Stimulus, mu='+str(np.round(np.mean(covSnonOS[np.triu_indices(covSnonOS.shape[0],k=0)])))])
    plt.xlabel(thisType+' covariance')
    plt.title('Non-OS only')


# Variances
if thisType=='full':
    meanVarOS = [np.round(np.mean(np.diag(covBOS)),1), np.round(np.mean(np.diag(covSOS)),1)]
    meanVarNonOS = [np.round(np.mean(np.diag(covBnonOS)),1), np.round(np.mean(np.diag(covSnonOS)),1)]
else:
    meanVarOS.append(np.round(np.mean(np.diag(covBOS)),1))
    meanVarOS.append(np.round(np.mean(np.diag(covSOS)),1))
    meanVarNonOS.append(np.round(np.mean(np.diag(covBnonOS)),1))
    meanVarNonOS.append(np.round(np.mean(np.diag(covSnonOS)),1))

# Covariances
if thisType=='full':
    meanCovOS = [np.round(np.mean(covBOS[np.triu_indices(covBOS.shape[0],k=0)]),1), np.round(np.mean(covSOS[np.triu_indices(covSOS.shape[0],k=0)]),1)]
    meanCovNonOS = [np.round(np.mean(covBnonOS[np.triu_indices(covBnonOS.shape[0],k=0)]),1), np.round(np.mean(covSnonOS[np.triu_indices(covSnonOS.shape[0],k=0)]),1)]
elif thisType=='shared':
    meanCovOS.append(np.round(np.mean(covBOS[np.triu_indices(covBOS.shape[0],k=0)]),1))
    meanCovOS.append(np.round(np.mean(covSOS[np.triu_indices(covSOS.shape[0],k=0)]),1))
    meanCovNonOS.append(np.round(np.mean(covBnonOS[np.triu_indices(covBnonOS.shape[0],k=0)]),1))
    meanCovNonOS.append(np.round(np.mean(covSnonOS[np.triu_indices(covSnonOS.shape[0],k=0)]),1))
    

plt.figure()
plt.plot(np.arange(6),meanVarOS)
plt.xticks(np.arange(6),labels=['full B','full S','shared B','shared S','private B','private S'])
plt.ylabel('Mean Var')
plt.title('OS only')

plt.figure()
plt.plot(np.arange(6),meanVarNonOS)
plt.xticks(np.arange(6),labels=['full B','full S','shared B','shared S','private B','private S'])
plt.ylabel('Mean Var')
plt.title('Non-OS only')

plt.figure()
plt.plot(np.arange(4),meanCovOS)
plt.xticks(np.arange(4),labels=['full B','full S','shared B','shared S'])
plt.ylabel('Mean Cov')
plt.title('OS only')

plt.figure()
plt.plot(np.arange(4),meanCovNonOS)
plt.xticks(np.arange(4),labels=['full B','full S','shared B','shared S'])
plt.ylabel('Mean Cov')
plt.title('Non-OS only')


#%%

### Check if OS ROIs code for different orientations using the same or different axes/dimensions ###
idxOS45 = np.where((np.array(charROI['OS'])) & (np.array(charROI['PrefOri'])==45))[0]
idxOS135 = np.where((np.array(charROI['OS'])) & (np.array(charROI['PrefOri'])==135))[0]
idxOS180 = np.where((np.array(charROI['OS'])) & (np.array(charROI['PrefOri'])==180))[0]
idxOS270 = np.where((np.array(charROI['OS'])) & (np.array(charROI['PrefOri'])==270))[0]

idxS45 = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==45))[0]
idxS135 = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==135))[0]
idxS180 = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==180))[0]
idxS270 = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==270))[0]

thisFluo = fluo[:,idxOS45][idxStimulus,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS270][idxStimulus,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20],nTimes=20,type='full',boolPlot=True)

thisFluo = np.concatenate((fluo[:,idxOS45][idxStimulus,:],fluo[:,idxOS270][idxStimulus,:]),axis=1)
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20],nTimes=20,type='full',boolPlot=True)


## Dimensionality of ROIs selective for 45° during different types of frames
thisFluo = fluo[:,idxOS45][idxS45,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS45)],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS45][idxS135,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS45)],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS45][idxS180,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS45)],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS45][idxS270,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS45)],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS45][idxBlank,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS45)],nTimes=20,type='full',boolPlot=True)


## Dimensionality of ROIs selective for 135° during different types of frames
thisFluo = fluo[:,idxOS135][idxS45,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS135)],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS135][idxS135,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS135)],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS135][idxS180,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS135)],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS135][idxS270,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS135)],nTimes=20,type='full',boolPlot=True)

thisFluo = fluo[:,idxOS135][idxBlank,:]
allPR = functions_analyze.compute_dimensionalityBootstrap(thisFluo,nROIs=[10,20,len(idxOS135)],nTimes=20,type='full',boolPlot=True)













