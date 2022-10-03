# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:22:31 2021

@author: Olivia Gozel

Analyze the thalamic boutons to L2/3 data with the behavioral data (if present)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze
import functions_postprocess

# Save in matlab format:
# from scipy import io
# io.savemat('filename.mat', {"boutons": np.array(fluo) })



#%% Parameters of the data to preprocess

### TO CHOOSE ###
dataType = 'L23_thalamicBoutons' # 'L23_thalamicBoutons'
# 'L4_LGN_targeted_axons'

filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')

### TO CHOOSE ###
idxDataset = 0 # 'L23_thalamicBoutons': 0, 2, 3, 5, 6

dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']

### To CHOOSE ###
dataNeuropilSub = globalParams.neuropilSub[0]



#%% Load data

filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'

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


# Save in matlab format:
# from scipy import io
# fluo_blank = fluo.loc[charTrials['FrameType']=='Blank'].reset_index(drop=True)
# fluo_stim = fluo.loc[charTrials['FrameType']=='Stimulus'].reset_index(drop=True)
# fluo_stim45 = fluo.loc[(charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==45)].reset_index(drop=True)
# fluo_stim45 = fluo_stim45.loc[0:4999]
# io.savemat('dataset6_boutons_L23_perType.mat', {"L23_blank": np.array(fluo_blank), "L23_stim": np.array(fluo_stim), "L23_stim45": np.array(fluo_stim45) })



#%% Plot ROIs with their selectivity

functions_postprocess.plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)

percROI_OS = 100*(charROI[charROI['OS']==True].shape[0] / charROI.shape[0])
print(str(np.around(percROI_OS))+'% thalamic boutons are orientation-selective (OS)')

# # L2/3 ROIs and thalamic boutons
# functions_preprocess.plot_ROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight)



#%% Plot the pairwise noise correlations as a function of pairwise distance

# OS ROIs only, only stimulus frames
binCenters, sortedPairCorr = functions_analyze.plot_noiseCorr_fdist(fluo,distROI,charROI,charTrials,\
                        boolVisuallyEvoked=True,boolOS=True,boolIdenticalTuning=False,boolStimulus=True,\
                            startBin=5,binSize=5)

# OS ROIs only, only blank frames
binCenters, sortedPairCorr = functions_analyze.plot_noiseCorr_fdist(fluo,distROI,charROI,charTrials,\
                        boolVisuallyEvoked=True,boolOS=True,boolIdenticalTuning=False,boolStimulus=False,\
                            startBin=5,binSize=5)


# idxOS = np.squeeze(np.array(np.where(np.array(charROI['OS']))))
# thisDistROI = np.array(distROI)[idxOS,:][:,idxOS]

# # Evoked frames (OS neurons only)
# idxStim = np.squeeze(np.array(np.where(charTrials['FrameType']=='Stimulus')))
# thisFluo = np.array(fluo)[idxStim,:][:,idxOS]

# binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='Thalamic boutons - Evoked - OS only')


# # Spontaneous frames (OS neurons only)
# idxBlank = np.squeeze(np.array(np.where(charTrials['FrameType']=='Blank')))
# thisFluo = np.array(fluo)[idxBlank,:][:,idxOS]

# binCenters, sortedPairCorr = functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='Thalamic boutons - Spontaneous - OS only')



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
charAxons = pd.DataFrame(index=np.arange(nAxons),columns=['nBoutons','boutonIdx','pairCorr','pairCorrWithOtherClusters',\
                                                          'pairDist','pairDistWithOtherClusters',\
                                                              'pairCorrBlank','pairCorrStimulus'])

# Compute the mean pairwise Pearson correlation within each cluster and between each cluster
for a in range(0,nAxons):
    
    # Within-cluster - only "events" within blank frames (see above)
    tmpidx = np.where(cutree==a)[0]
    thisNBoutons = len(tmpidx) # number of boutons within the cluster
    tmpFullPearsonCorr = np.array(corr_matrix_full.iloc[tmpidx][tmpidx]) # only the pairwise correlations between boutons of the same cluster
    tmpPearsonCorr = tmpFullPearsonCorr[np.triu_indices(thisNBoutons, k = 1)] # pairwise corr between boutons assigned to the same cluster
    tmpPairDist = np.array(distROI)[tmpidx,:][:,tmpidx]
    tmpPairDist = tmpPairDist[np.triu_indices(thisNBoutons, k = 1)]
    
    # Between clusters - only "events" within blank frames (see above)
    tmpNONidx = np.setdiff1d(np.arange(nROI),tmpidx)
    tmpBtwPearsonCorr = np.array(corr_matrix_full.iloc[tmpidx][tmpNONidx]) # only the pairwise correlations between boutons belonging to different clusters
    tmpBtwPairDist = np.array(distROI)[tmpidx,:][:,tmpNONidx]
    
    # Within-cluster noise correlations for all the blank frames
    # (not only the events within the blank frames as above)
    tmpWithinBlank = fluo[tmpidx].loc[charTrials['FrameType']=='Blank']
    tmpCorrBlank = np.array(tmpWithinBlank.corr())[np.triu_indices(thisNBoutons, k = 1)]
    
    # # Between-cluster noise correlations for all the blank frames
    # tmpBetweenBlank = fluo[np.union1d(tmpidx,tmpNONidx)].loc[charTrials['FrameType']=='Blank']
    # tmpCorrBtwBlank = np.mean(np.array(tmpBetweenBlank.corr())[tmpidx,:][:,tmpNONidx])
    
    # Within-cluster and between-cluster noise correlations for all the stimulus frames
    # Noise correlations, so we need to take trials of each orientation separately
    tmpCorrStim = []
    # tmpCorrBtwStim = []
    for o in range(globalParams.nOri):
        theseFrames = np.where((charTrials['FrameType']=='Stimulus')&(charTrials['Orientation']==globalParams.ori[o]))[0]
        tmpWithinStim = fluo[tmpidx].loc[theseFrames]
        tmpCorrStim.append(np.array(tmpWithinStim.corr())[np.triu_indices(thisNBoutons, k = 1)])
        # tmpBtwStim = fluo[np.union1d(tmpidx,tmpNONidx)].loc[theseFrames]
        # tmpCorrBtwStim.append(np.mean(np.array(tmpBtwStim.corr())[tmpidx,:][:,tmpNONidx]))
    tmpCorrStim = np.array(tmpCorrStim).flatten()
    # tmpCorrBtwStim = np.array(tmpCorrBtwStim).flatten()
    

    
    # Write the data in the dataframe
    charAxons.iloc[a]['nBoutons'] = thisNBoutons
    charAxons.iloc[a]['boutonIdx'] = tmpidx
    charAxons.iloc[a]['pairCorr'] = tmpPearsonCorr
    charAxons.iloc[a]['pairCorrWithOtherClusters'] = np.concatenate(tmpBtwPearsonCorr)
    charAxons.iloc[a]['pairDist'] = tmpPairDist
    charAxons.iloc[a]['pairDistWithOtherClusters'] = np.concatenate(tmpBtwPairDist)
    charAxons.iloc[a]['pairCorrBlank'] = tmpCorrBlank
    charAxons.iloc[a]['pairCorrStimulus'] = tmpCorrStim
    # charAxons.iloc[a]['pairCorrBtwBlank'] = tmpCorrBtwBlank
    # charAxons.iloc[a]['pairCorrBtwStimulus'] = tmpCorrBtwStim


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

# Plot the normalized distribution of pairwise correlation within-cluster and between clusters (normalized such that the area under the curve is equal to 1)
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

# Plot the normalized distribution of pairwise correlation within each putative axon for spontaneous versus evoked
plt.figure()
pairCorrWithinBlank = np.concatenate(charAxons['pairCorrBlank'])
pairCorrWithinStim = np.concatenate(charAxons['pairCorrStimulus'])
plt.hist(pairCorrWithinBlank,bins=np.arange(-0.3,0.3+0.05,0.05),density=True,alpha = 0.5)
plt.hist(pairCorrWithinStim,bins=np.arange(-0.3,0.3+0.05,0.05),density=True,alpha=0.5)
plt.xlabel('Pairwise correlations')
plt.ylabel('Density')
plt.legend(['Spontaneous','Evoked'])

# Plot the average pairwise noise correlation over each putative axon for blank frames versus stimulus frames
f, ax = plt.subplots()
plt.scatter(charAxons['pairCorrBlank'].apply(np.mean),charAxons['pairCorrStimulus'].apply(np.mean))
plt.axline((0, 0), slope=1, color="black") # diagonal (unity) line
# ax.set_aspect(aspect=1) # if we want to keep the length on both axes the same
plt.xlabel('Avg noise corr per bouton cluster during blank frames')
plt.ylabel('Avg noise corr per bouton cluster during stimulus frames')


#%% Compute Pearson correlation between 1st motion PC and 1st neural PC

idxOS = np.where(np.array(charROI['OS']))[0]
idxNonOS = np.where(np.array(charROI['OS']==False))[0]
idxBlank = np.where(charTrials['FrameType']=='Blank')[0]
idxStimulus = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==45))[0] ### !!!

if len(idxBlank) < len(idxStimulus):
    idxStimulus = idxStimulus[0:len(idxBlank)]

motSVD1_blank = charTrials['motSVD1'].iloc[idxBlank]
motSVD1_stim = charTrials['motSVD1'].iloc[idxStimulus]

fluoOS_blank = fluo[idxOS].iloc[idxBlank]
neurProj_OS_blank,_ = functions_analyze.compute_neuralPCs(fluoOS_blank)
fluoOS_stim = fluo[idxOS].iloc[idxStimulus]
neurProj_OS_stim,_ = functions_analyze.compute_neuralPCs(fluoOS_stim)
fluoNonOS_blank = fluo[idxNonOS].iloc[idxBlank]
neurProj_nonOS_blank,_ = functions_analyze.compute_neuralPCs(fluoNonOS_blank)
fluoNonOS_stim = fluo[idxNonOS].iloc[idxStimulus]
neurProj_nonOS_stim,_ = functions_analyze.compute_neuralPCs(fluoNonOS_stim)

print('Pearson corr neuralPC1-motionPC1 spontaneous OS: '+str(np.corrcoef(motSVD1_blank,neurProj_OS_blank[:,0])[0,1]))
print('Pearson corr neuralPC1-motionPC1 evoked OS: '+str(np.corrcoef(motSVD1_stim,neurProj_OS_stim[:,0])[0,1]))
print('Pearson corr neuralPC1-motionPC1 spontaneous nonOS: '+str(np.corrcoef(motSVD1_blank,neurProj_nonOS_blank[:,0])[0,1]))
print('Pearson corr neuralPC1-motionPC1 evoked nonOS: '+str(np.corrcoef(motSVD1_stim,neurProj_nonOS_stim[:,0])[0,1]))


#%% Compute Pearson correlation between 1st neural PC of boutons and 1st neural PC of L2/3

dataNeuropilSubV1 = globalParams.neuropilSub[3]

filepathV1 = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSubV1 + '_threshDist2d5um.hdf'

fluoINITV1 = pd.read_hdf(filepathV1,'fluo')
charROIV1 = pd.read_hdf(filepathV1,'charROI')

idxOSV1 = np.where(np.array(charROIV1['OS']))[0]
idxNonOSV1 = np.where(np.array(charROIV1['OS']==False))[0]

fluoOS_blankV1 = fluo[idxOSV1].iloc[idxBlank]
neurProj_OS_blankV1,_ = functions_analyze.compute_neuralPCs(fluoOS_blankV1)
fluoOS_stimV1 = fluo[idxOSV1].iloc[idxStimulus]
neurProj_OS_stimV1,_ = functions_analyze.compute_neuralPCs(fluoOS_stimV1)
fluoNonOS_blankV1 = fluo[idxNonOSV1].iloc[idxBlank]
neurProj_nonOS_blankV1,_ = functions_analyze.compute_neuralPCs(fluoNonOS_blankV1)
fluoNonOS_stimV1 = fluo[idxNonOSV1].iloc[idxStimulus]
neurProj_nonOS_stimV1,_ = functions_analyze.compute_neuralPCs(fluoNonOS_stimV1)

print('Pearson corr neuralPC1-neuralPC1 spontaneous OS: '+str(np.corrcoef(neurProj_OS_blankV1[:,0],neurProj_OS_blank[:,0])[0,1]))
print('Pearson corr neuralPC1-neuralPC1 evoked OS: '+str(np.corrcoef(neurProj_OS_stimV1[:,0],neurProj_OS_stim[:,0])[0,1]))
print('Pearson corr neuralPC1-neuralPC1 spontaneous nonOS: '+str(np.corrcoef(neurProj_nonOS_blankV1[:,0],neurProj_nonOS_blank[:,0])[0,1]))
print('Pearson corr neuralPC1-neuralPC1 evoked nonOS: '+str(np.corrcoef(neurProj_nonOS_stimV1[:,0],neurProj_nonOS_stim[:,0])[0,1]))





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




