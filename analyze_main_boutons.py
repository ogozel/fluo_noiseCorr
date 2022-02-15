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



#%% Parameters of the data to preprocess

### TO CHOOSE ###
dataType = 'L23_thalamicBoutons' # 'L23_thalamicBoutons'

filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')

### TO CHOOSE ###
idxDataset = 3

dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']

### To CHOOSE ###
dataNeuropilSub = globalParams.neuropilSub[0]



#%% Load data

filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist2d5um.hdf'

# Thalamic bouton data
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


# if 'pupilArea' in charTrials:
#     bool_pupil = True
# else:
#     bool_pupil = False

# store = pd.HDFStore(filepath,mode='r')
# if 'motAvg' in store:
#     bool_motion = True
#     motAvg = pd.read_hdf(filepath,'motAvg')
#     uMotMask = pd.read_hdf(filepath,'uMotMask')
# else:
#     bool_motion = False




#%% Plot ROIs with their selectivity

functions_preprocess.plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)

percROI_OS = 100*(charROI[charROI['OS']==True].shape[0] / charROI.shape[0])
print(str(np.around(percROI_OS))+'% thalamic boutons are orientation-selective (OS)')

# # L2/3 ROIs and thalamic boutons
# functions_preprocess.plot_ROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight)



#%% Plot the pairwise correlation as a function of pairwise distance

idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
thisDistROI = np.array(distROI)[idxOS,:][:,idxOS]

# Evoked frames (OS neurons only)
idxStim = np.squeeze(np.array(np.where(charTrials['FrameType']=='Stimulus')))
thisFluo = np.array(fluo)[idxStim,:][:,idxOS]

binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='Thalamic boutons - Evoked - OS only')


# Spontaneous frames (OS neurons only)
idxBlank = np.squeeze(np.array(np.where(charTrials['FrameType']=='Blank')))
thisFluo = np.array(fluo)[idxBlank,:][:,idxOS]

binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='Thalamic boutons - Spontaneous - OS only')



#%% Link together thalamic boutons that arise from the same axon using a correlation-based hierarchical clustering procedure
# (Liang et al., 2018; STAR Methods)

# Pick only blank frames
tmpBF = np.where(charTrials['FrameType']=='Blank')[0]
fluoBF = fluo.iloc[tmpBF]

# Compute the mean and standard deviation
fluoBF_mean = fluoBF.mean()
fluoBF_std = fluoBF.std()

# 3 standard deviations above and below the mean
thresh_below = fluoBF_mean - fluoBF_std
thresh_above = fluoBF_mean + fluoBF_std

# mask_below = fluoBF[fluoBF<thresh_below]
# mask_above = fluoBF[fluoBF>thresh_above]
mask_below_above = fluoBF[(fluoBF<thresh_below) | (fluoBF>thresh_above)]

# Compute Pearson correlation for each pair of thalamic boutons (does not take into account nan values)
corr_matrix_full = mask_below_above.corr() # to keep all the pairwise correlation values
corr_matrix = mask_below_above.corr() # used and modified later to do the clustering

# Set diagonal to nan
np.fill_diagonal(corr_matrix.values, np.nan)

# # Find pairwise correlations higher than 0.7
# idx_higherThan0d7 = np.where(corr_matrix>0.7)

# Find pairwise correlations higher than 2.5 standard deviations above the mean value of all the coefficients between this bouton and all others
corr_meanPerBouton = corr_matrix.mean()
corr_stdPerBouton = corr_matrix.std()
thresh2 = corr_meanPerBouton+2.5*corr_stdPerBouton
# idx_higherThan2d5stdMean = np.where(corr_matrix>thresh2)

# Set pairwise correlation coefficients to 0 if they do not exceed 0.7 or 2.5 std above the mean corr between this bouton and all others
# corr_matrix_thresh = corr_matrix
# corr_matrix_thresh[(corr_matrix_thresh<=0.7)&(corr_matrix_thresh<=thresh2)] = 0
corr_matrix[(corr_matrix<=0.7)&(corr_matrix<=thresh2)] = 0

# Set diagonal back to 1
np.fill_diagonal(corr_matrix.values, 1.0)

# Set nan elements to 0
corr_matrix[corr_matrix.isnull()] = 0

# Compute cosyne similarity between each pair of boutons
from sklearn.metrics.pairwise import cosine_similarity
cosyneSim = cosine_similarity(corr_matrix)

# Compute pairwise distance, defined as ‘1 – cosine similarity’
pairDist = 1 - cosyneSim
np.fill_diagonal(pairDist, 0.0) # take care of approximation errors

# Perform hierarchy clustering using the weighted-pair group method with arithmetic means (WPGMA) algorithm
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
condensedPairDist = squareform(pairDist)

thisClustering = linkage(condensedPairDist, method='weighted')

from scipy.cluster.hierarchy import cut_tree
cutree = cut_tree(thisClustering, height=0.85)

# Number of clusters (= putative axons)
nAxons = np.max(cutree)


#%% Analyze the clusters of thalamic boutons (=putative axons)

# Create a dataframe with the info concerning the clusters of boutons
charAxons = pd.DataFrame(index=np.arange(nAxons),columns=['nBoutons','pairCorr','boutonIdx','pairCorrWithOtherClusters',\
                                                          'pairDist','pairDistWithOtherClusters'])

# Compute the mean pairwise Pearson correlation within each cluster and between each cluster
for a in range(0,nAxons):
    # Within-cluster
    tmpidx = np.where(cutree==a)[0]
    thisNBoutons = len(tmpidx) # number of boutons within the cluster
    tmpFullPearsonCorr = np.array(corr_matrix_full.iloc[tmpidx][tmpidx]) # only the pairwise correlations between boutons of the same cluster
    tmpPearsonCorr = tmpFullPearsonCorr[np.triu_indices(thisNBoutons, k = 1)] # pairwise corr between boutons assigned to the same cluster
    tmpPairDist = np.array(distROI)[tmpidx,:][:,tmpidx]
    tmpPairDist = tmpPairDist[np.triu_indices(thisNBoutons, k = 1)]
    
    # Between clusters
    tmpNONidx = np.setdiff1d(np.arange(nROI),tmpidx)
    tmpBtwPearsonCorr = np.array(corr_matrix_full.iloc[tmpidx][tmpNONidx]) # only the pairwise correlations between boutons belonging to different clusters
    tmpBtwPairDist = np.array(distROI)[tmpidx,:][:,tmpNONidx]
    
    # Write the data in the dataframe
    charAxons.iloc[a]['nBoutons'] = thisNBoutons
    charAxons.iloc[a]['pairCorr'] = tmpPearsonCorr
    charAxons.iloc[a]['boutonIdx'] = tmpidx
    charAxons.iloc[a]['pairCorrWithOtherClusters'] = np.concatenate(tmpBtwPearsonCorr)
    charAxons.iloc[a]['pairDist'] = tmpPairDist
    charAxons.iloc[a]['pairDistWithOtherClusters'] = np.concatenate(tmpBtwPairDist)


# Plot the distribution of number of boutons per axon
plt.figure()
charAxons['nBoutons'].hist(bins=np.arange(0.5,1.5+charAxons['nBoutons'].max()),grid=False)
plt.xlabel('Number of boutons per cluster')
#plt.xticks(np.arange(1,1+charAxons['nBoutons'].max()))
plt.ylabel('Nunber of counts')


# Plot the normalized distribution of pairwise correlation within-cluster (normalized such that the area under the curve is equal to 1)
plt.figure()
pairCorrWithin = np.concatenate(charAxons['pairCorr'])
plt.hist(pairCorrWithin,bins=np.arange(-1,1+0.1,0.1),density=True)
plt.xlabel('Pairwise correlation within cluster')
plt.ylabel('Density')

# Plot the normalized distribution of pairwise correlation between clusters (normalized such that the area under the curve is equal to 1)
plt.figure()
pairCorrBtw = np.concatenate(charAxons['pairCorrWithOtherClusters'])
plt.hist(pairCorrBtw,bins=np.arange(-1,1+0.1,0.1),density=True)
plt.xlabel('Pairwise correlation between clusters')
plt.ylabel('Density')

# Plot the normalized distribution of pairwise correlation within-cluster and between clusters
plt.figure()
plt.hist(pairCorrWithin,bins=np.arange(-1,1+0.1,0.1),density=True,alpha = 0.5)
plt.hist(pairCorrBtw,bins=np.arange(-1,1+0.1,0.1),density=True,alpha=0.5)
plt.xlabel('Pairwise correlations')
plt.ylabel('Density')
plt.legend(['Within-cluster','Between clusters'])

# Plot the pairwise distance within-cluster and between clusters
plt.figure()
plt.hist(np.concatenate(charAxons['pairDist']),bins=np.arange(0,500,50),density=True,alpha = 0.5)
plt.hist(np.concatenate(charAxons['pairDistWithOtherClusters']),bins=np.arange(0,500,50),density=True,alpha=0.5)
plt.xlabel('Pairwise distances')
plt.ylabel('Density')
plt.legend(['Within-cluster','Between clusters'])






#%% Compute the neural PCs

# Select the ROIs we are interested in
idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
#thisFluo = np.array(fluo)
thisFluo = np.array(fluo)[:,idxOS]

# All frames, only OS neurons
neuralProj_boutons, percVarExpl_boutons = functions_analyze.compute_neuralPCs(thisFluo,bool_plot=1,title='Boutons - All frames - OS only')



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




