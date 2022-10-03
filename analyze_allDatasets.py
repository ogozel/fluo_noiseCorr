# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:55:32 2022

@author: Olivia Gozel

Script to do summary statistics over all datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze




#%% Parameters of the preprocessed data to analyze

### TO CHOOSE ###
dataType = 'L23_thalamicBoutons' # 'L4_cytosolic' or 'L23_thalamicBoutons'
boolV1 = True # True or False
numDatasets = np.arange(8)
# np.arange(10) for V1 L4
# np.arange(8) for V1 L2/3
# np.array((0,2,3,5,6)) for L2/3 thalamic boutons
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor
# globalParams.neuropilSub[3] for V1 data
# globalParams.neuropilSub[0] for thalamic boutons

boolPR = True


filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')

# Determine what is plotted
if dataType == 'L23_thalamicBoutons':
    if boolV1 == True:
        dataName = 'L2/3'
    else:
        dataName = 'Boutons to L2/3'
else:
    dataName = dataType #################!!! Ok for now but probably need to change later


# %% Determine the max number of blank frames and stimulus frames in the datasets of interest

nBlankFrames_max = 0
nStimFrames_max = 0

# Loop over all datasets
for dd in range(len(numDatasets)):
    
    idxDataset = numDatasets[dd]
    
    nBlankFrames = dataSpecs.iloc[idxDataset]['nBlankFrames']
    nStimFrames = dataSpecs.iloc[idxDataset]['nStimFrames']
    
    if nBlankFrames > nBlankFrames_max:
        nBlankFrames_max = nBlankFrames
    if nStimFrames > nStimFrames_max:
        nStimFrames_max = nStimFrames
        
# Maximal total number of frames per trial
dataFramesPerTrial_max = nBlankFrames_max + nStimFrames_max


#%% Average fluorescence and coefficient of variation

# Initialize the matrices
master_mean = np.empty((dataFramesPerTrial_max,len(numDatasets)))
master_mean[:] = np.nan
master_std = np.empty((dataFramesPerTrial_max,len(numDatasets)))
master_std[:] = np.nan

# Loop over all datasets
for dd in range(len(numDatasets)):
    
    ##########################################################################
    # Load data
    ##########################################################################
    
    idxDataset = numDatasets[dd]
    
    dataDate = dataSpecs.iloc[idxDataset]['Date']
    dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
    dataDepth = dataSpecs.iloc[idxDataset]['Depth']
    pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
    dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
    dataFR = dataSpecs.iloc[idxDataset]['FrameRate']
    
    nBlankFrames = dataSpecs.iloc[idxDataset]['nBlankFrames']
    nStimFrames = dataSpecs.iloc[idxDataset]['nStimFrames']
    dataFramesPerTrial = nBlankFrames + nStimFrames
    
    if boolV1:
        filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
                dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    else:
        filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
            dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    
    fluo = pd.read_hdf(filepath,'fluo')
    charTrials = pd.read_hdf(filepath,'charTrials')

    # functions_analyze.plot_avgFluo(dataFramesPerTrial,charTrials,fluo,dataFR)
    
    thisCharTrials = charTrials[['Orientation','TrialFrame']]
    data = pd.concat([thisCharTrials, fluo],axis=1)
    
    avgPerOri = data.groupby(['TrialFrame','Orientation']).mean()
    avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
    avgPerOri = avgPerOri.reset_index()

    tmp = avgPerOri
    tmp = np.array(tmp.drop(['TrialFrame','Orientation'],axis=1))
    tmp = np.reshape(tmp,(dataFramesPerTrial,-1),order='F')
    tmpMean = np.mean(tmp,axis=1)
    tmpStd = np.std(tmp,axis=1)
    
    diffB = nBlankFrames_max - nBlankFrames
    master_mean[diffB:diffB+dataFramesPerTrial,dd] = tmpMean
    master_std[diffB:diffB+dataFramesPerTrial,dd] = tmpStd
    
    # if dd==0:
    #     master_mean = tmpMean[:,np.newaxis]
    #     master_std = tmpStd[:,np.newaxis]
    # else:
    #     if len(tmpMean) < len(master_mean): # hack for the L2/3 datasets that have less frames per trial
    #         tmpMean = np.concatenate((tmpMean,np.repeat(np.nan, len(master_mean)-len(tmpMean))))
    #         tmpStd = np.concatenate((tmpStd,np.repeat(np.nan, len(master_std)-len(tmpStd))))
    #     master_mean = np.concatenate((master_mean,tmpMean[:,np.newaxis]),axis=1)
    #     master_std = np.concatenate((master_std,tmpStd[:,np.newaxis]),axis=1)


tmpTime = np.arange(len(master_mean))-nBlankFrames_max
plt.figure()
for dd in range(len(numDatasets)):
    plt.plot(tmpTime, master_std[:,dd]/master_mean[:,dd],label=str(numDatasets[dd]))
plt.xlabel('Frame aligned to stimulus onset')
plt.ylabel('Coefficient of variation: std/mean')
plt.title(dataName)
plt.legend()

plt.figure()
for dd in range(len(numDatasets)):
    #plt.fill_between(tmpTime,master_mean[:,dd]-master_std[:,dd],master_mean[:,dd]+master_std[:,dd])
    plt.plot(tmpTime, master_mean[:,dd],label=str(numDatasets[dd]))
plt.xlabel('Frame aligned to stimulus onset')
plt.ylabel('Average fluo')
plt.title(dataName)
plt.legend()

plt.figure()
for dd in range(len(numDatasets)):
    #plt.fill_between(tmpTime,master_mean[:,dd]-master_std[:,dd],master_mean[:,dd]+master_std[:,dd])
    plt.plot(tmpTime, master_std[:,dd],label=str(numDatasets[dd]))
plt.xlabel('Frame aligned to stimulus onset')
plt.ylabel('Std of fluo')
plt.title(dataName)
plt.legend()

#%% Distribution and PR of correlations or covariances

cMatType = 'correlationMatrix' # 'covarianceMatrix' or 'correlationMatrix'

from scipy.stats import ks_2samp
from scipy.stats import ttest_ind

Master_dim = pd.DataFrame(columns=['spont','evoked'])
#Master_eigVal = pd.DataFrame(columns=['spont','evoked'])
Master_var = pd.DataFrame(columns=['MeanSpont','MeanEvoked','MedianSpont','MedianEvoked'])
Master_cov = pd.DataFrame(columns=['MeanSpont','MeanEvoked','MedianSpont','MedianEvoked'])

# Loop over all datasets
for dd in range(len(numDatasets)):
    
    ##########################################################################
    # Load data
    ##########################################################################
    
    idxDataset = numDatasets[dd]
    
    dataDate = dataSpecs.iloc[idxDataset]['Date']
    dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
    dataDepth = dataSpecs.iloc[idxDataset]['Depth']
    pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
    dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
    dataFR = dataSpecs.iloc[idxDataset]['FrameRate']
    
    nBlankFrames = dataSpecs.iloc[idxDataset]['nBlankFrames']
    nStimFrames = dataSpecs.iloc[idxDataset]['nStimFrames']
    dataFramesPerTrial = nBlankFrames + nStimFrames
    
    if boolV1:
        filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
                dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'    
    else:
        filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
            dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    
    fluo = pd.read_hdf(filepath,'fluo')
    charROI = pd.read_hdf(filepath,'charROI')
    charTrials = pd.read_hdf(filepath,'charTrials')
    
    # Correlation matrix
    fluo_blank = fluo.loc[charTrials['FrameType']=='Blank']
    if cMatType == 'covarianceMatrix':
        cMat_blank = np.array(fluo_blank.cov())
    elif cMatType == 'correlationMatrix':
        cMat_blank = np.array(fluo_blank.corr(method='pearson'))
    
    var_blank = np.diag(cMat_blank)
    c_blank = cMat_blank[np.triu_indices(cMat_blank.shape[0],k=1)]
    
    [u_blank, s_blank, vh] = np.linalg.svd(cMat_blank)
    thisPRoverN_blank = (np.power(np.sum(s_blank),2)/np.sum(np.power(s_blank,2)))/len(s_blank)
    
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
    figV, axV = plt.subplots(2, 2)
    figC, axC = plt.subplots(2, 2)
    figVarExpl, axsVarExpl = plt.subplots()
    print('Dataset'+str(idxDataset)+' Spont: '+str(np.mean(c_blank))+' +/- '+str(np.std(c_blank)))
    thisPRoverN = 0
    f, ax = plt.subplots()
    for o in range(globalParams.nOri):
        fluo_stim = fluo.loc[(charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==globalParams.ori[o])]
        
        if cMatType == 'covarianceMatrix':
            cMat_stim = np.array(fluo_stim.cov())
        elif cMatType == 'correlationMatrix':   
            cMat_stim = np.array(fluo_stim.corr(method='pearson'))
        var_stim = np.diag(cMat_stim)
        
        [u_stim, s_stim, vh] = np.linalg.svd(cMat_stim)
        thisPRoverN += (np.power(np.sum(s_stim),2)/np.sum(np.power(s_stim,2)))/len(s_stim)

        c_stim = cMat_stim[np.triu_indices(cMat_stim.shape[0],k=1)]
        bins = np.histogram(np.hstack((c_blank,c_stim)), bins=40)[1] #get the bin edges
        axs[int(o/2), np.mod(o,2)].hist(c_blank,bins,alpha=0.5)
        axs[int(o/2), np.mod(o,2)].hist(c_stim,bins,alpha=0.5)
        axs[int(o/2), np.mod(o,2)].set_title('Orientation: '+str(globalParams.ori[o])+'°')
        print(str(globalParams.ori[o])+'°: '+str(np.mean(c_stim))+' +/- '+str(np.std(c_stim)))
        print('KS test: '+str(ks_2samp(c_blank, c_stim, alternative='two-sided')))
        print('Two sample t-test: '+str(ttest_ind(c_blank, c_stim, equal_var=True)))
        print('Welsch test: '+str(ttest_ind(c_blank, c_stim, equal_var=False)))
        
        axV[int(o/2), np.mod(o,2)].hist(var_blank-var_stim,40)
        axC[int(o/2), np.mod(o,2)].hist(c_blank-c_stim,40)
        
        axsVarExpl.scatter(s_blank/np.sum(s_blank), s_stim/np.sum(s_stim))
        
        ax.scatter(s_blank,s_stim)
    
    fig.suptitle(cMatType)
    
    figV.suptitle('Variances spont-evoked: '+dataName+', dataset'+str(idxDataset))
    figC.suptitle('Covariances spont-evoked: '+dataName+', dataset'+str(idxDataset))
    ax.set_title(dataName+', dataset'+str(idxDataset))
    
    axsVarExpl.plot([0, np.max(axsVarExpl.get_xlim())], [0, np.max(axsVarExpl.get_xlim())], color='k')
    axsVarExpl.set_xlabel('EigV/sumEigV spont')
    axsVarExpl.set_ylabel('EigV/sumEigV evoked')
    axsVarExpl.set_title(dataName+', dataset'+str(idxDataset))
    
    ax.plot([0, np.max(np.ceil(ax.get_xlim()))], [0, np.max(np.ceil(ax.get_xlim()))], color='k')
    ax.set_xlabel('EigV spont')
    ax.set_ylabel('EigV evoked')
    ax.set_title(dataName+', dataset'+str(idxDataset))
    
    Master_dim = Master_dim.append({'spont' : thisPRoverN_blank, 'evoked' : thisPRoverN/4},ignore_index = True)
    Master_var = Master_var.append({'MeanSpont' : np.mean(var_blank), 'MeanEvoked' : np.mean(var_stim), \
                                    'MedianSpont' : np.median(var_blank), 'MedianEvoked' : np.median(var_stim)},ignore_index = True)
    Master_cov = Master_cov.append({'MeanSpont' : np.mean(c_blank), 'MeanEvoked' : np.mean(c_stim), \
                                    'MedianSpont' : np.median(c_blank), 'MedianEvoked' : np.median(c_stim)},ignore_index = True)


x = np.arange(len(numDatasets))
width = 0.2  # the width of the bars
    
tmp1 = Master_dim['spont']
tmp2 = Master_dim['evoked']
fig, ax = plt.subplots()
ax.bar(x - width/2, tmp1, width, label='Spont')
ax.bar(x + width/2, tmp2, width, label='Evoked')
ax.set_ylabel(' PR / N')
ax.set_title(dataName+' - PR of '+cMatType)
ax.legend()

z = Master_dim['spont']-Master_dim['evoked']
zNorm = (Master_dim['spont']-Master_dim['evoked'])/Master_dim['evoked']

plt.figure()
plt.hist(z)
plt.title(dataName+' - PR_spont-PR_evoked')

plt.figure()
plt.hist(zNorm)
plt.title(dataName+' - (PR_spont-PR_evoked)/PR-evoked')

# savefilepath = globalParams.processedDataDir +'_dimCorr_allDatasets.hdf'
# Master_dimCorr_L4.to_hdf(savefilepath,key='Master_dimCorr_L4')
# Master_dimCorr_L23.to_hdf(savefilepath,key='Master_dimCorr_L23')
# Master_dimCorr_L23boutons.to_hdf(savefilepath,key='Master_dimCorr_L23boutons')


#%% Dimensionality NEW !!!

# Initialize dataframes where to save the dimensionality info
Master_dim_blank_OS = pd.DataFrame(columns=['nROIs','PR','PR_rate'])
Master_dim_stim_OS = pd.DataFrame(columns=['nROIs','PR','PR_rate'])
Master_dim_blank_nonOS = pd.DataFrame(columns=['nROIs','PR','PR_rate'])
Master_dim_stim_nonOS = pd.DataFrame(columns=['nROIs','PR','PR_rate'])


# Loop over all datasets
for dd in range(len(numDatasets)):
    
    ##########################################################################
    # Load data
    ##########################################################################
    
    idxDataset = numDatasets[dd]
    
    dataDate = dataSpecs.iloc[idxDataset]['Date']
    dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
    dataDepth = dataSpecs.iloc[idxDataset]['Depth']
    pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
    dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
    
    if boolV1:
        filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
                dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    else:
        filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
            dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    
    fluoDF = pd.read_hdf(filepath,'fluo')
    charROI = pd.read_hdf(filepath,'charROI')
    charTrials = pd.read_hdf(filepath,'charTrials')
    positionROI = pd.read_hdf(filepath,'positionROI')
    distROI = pd.read_hdf(filepath,'distROI')
    fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
    fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
    fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()
    
    nROI_init = fluoDF.shape[1]
    fluoDF.loc[np.where((fluoDF>1000)|(fluoDF<-1000))[0]] = np.nan
    varDF = fluoDF.var()
    plt.figure()
    plt.hist(varDF)
    plt.title('Variability of ROIs (before discarding)')
    
    idxTooVar = np.where(varDF>np.percentile(varDF, 80))[0] # np.where(varDF>np.percentile(varDF, 80))[0]
    print('There are '+str(len(idxTooVar))+'/'+str(nROI_init)+' ROIs which are highly variable')
    fluoDF = fluoDF.drop(columns=idxTooVar)
    fluoDF.columns = range(fluoDF.shape[1]) # reset the ROIs indices
    charROI = charROI.iloc[np.setdiff1d(np.arange(charROI.shape[0]),idxTooVar)].reset_index()
    varDF = fluoDF.var()
    
    plt.figure()
    plt.hist(varDF)
    plt.title('Variability of ROIs (after discarding)')
    
    # Select the ROIs and frame type we are interested in
    idxOS = np.where(np.array(charROI['OS']))[0]
    idxNonOS = np.where(np.array(charROI['OS']==False))[0]
    idxBlank = np.where(charTrials['FrameType']=='Blank')[0]
    #idxStimulus = np.where(charTrials['FrameType']=='Stimulus')[0] ### !!!
    idxStimulus = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==45))[0] ### !!!
    ### !!! len(idxStimulus) is not always = len(idxBlank)
    
    # OS only, blank frames
    prBOS,nOS = functions_analyze.compute_dimensionalityDF(fluoDF[idxOS].iloc[idxBlank])
    frMean = np.array(fluoDF[idxOS].iloc[idxBlank].mean())
    prBOS_rate = np.power(np.sum(frMean),2)/np.sum(np.power(frMean,2))
    
    # OS only, stimulus frames
    prSOS,nOS = functions_analyze.compute_dimensionalityDF(fluoDF[idxOS].iloc[idxStimulus])
    frMean = np.array(fluoDF[idxOS].iloc[idxStimulus].mean())
    prSOS_rate = np.power(np.sum(frMean),2)/np.sum(np.power(frMean,2))
    
    # Non-OS only, blank frames
    prBnonOS,nNonOS = functions_analyze.compute_dimensionalityDF(fluoDF[idxNonOS].iloc[idxBlank])
    frMean = np.array(fluoDF[idxNonOS].iloc[idxBlank].mean())
    prBnonOS_rate = np.power(np.sum(frMean),2)/np.sum(np.power(frMean,2))
    
    # Non-OS only, stimulus frames
    prSnonOS,nNonOS = functions_analyze.compute_dimensionalityDF(fluoDF[idxNonOS].iloc[idxStimulus])
    frMean = np.array(fluoDF[idxNonOS].iloc[idxStimulus].mean())
    prSnonOS_rate = np.power(np.sum(frMean),2)/np.sum(np.power(frMean,2))
    
    Master_dim_blank_OS.loc[dd, ['nROIs','PR','PR_rate']] = nOS, prBOS, prBOS_rate
    Master_dim_stim_OS.loc[dd, ['nROIs','PR','PR_rate']] = nOS, prSOS, prSOS_rate
    Master_dim_blank_nonOS.loc[dd, ['nROIs','PR','PR_rate']] = nNonOS, prBnonOS, prBnonOS_rate
    Master_dim_stim_nonOS.loc[dd, ['nROIs','PR','PR_rate']] = nNonOS, prSnonOS, prSnonOS_rate


plt.figure()
#plt.scatter(Master_dim_blank_OS['PR_rate'],Master_dim_blank_OS['PR'])
plt.scatter(Master_dim_stim_OS['PR_rate']/Master_dim_stim_OS['nROIs'],Master_dim_stim_OS['PR']/Master_dim_stim_OS['nROIs'])
#plt.scatter(Master_dim_blank_nonOS['PR_rate'],Master_dim_blank_nonOS['PR'])
plt.scatter(Master_dim_stim_nonOS['PR_rate']/Master_dim_stim_nonOS['nROIs'],Master_dim_stim_nonOS['PR']/Master_dim_stim_nonOS['nROIs'])
plt.xlabel('PR(mean dff0)/N')
plt.ylabel('PR(C)/N')
#plt.legend({'Blank, OS','Stim45, OS','Blank, nonOS','Stim45, nonOS'})
plt.legend({'Stim45, OS','Stim45, nonOS'})
plt.xlim([0, 1])
plt.xlim([0, 1])
plt.gca().set_aspect('equal')


#%% Dimensionality

# Initialize dataframes where to save the dimensionality info
Master_dim_blank_OS = pd.DataFrame(columns=['nTooVar','nROIs','PR','sumLambdas','lambda1','lambda2','lambda3','lambda4','meanVar','meanCov'])
Master_dim_stim_OS = pd.DataFrame(columns=['nTooVar','nROIs','PR','sumLambdas','lambda1','lambda2','lambda3','lambda4','meanVar','meanCov'])
Master_dim_blank_nonOS = pd.DataFrame(columns=['nTooVar','nROIs','PR','sumLambdas','lambda1','lambda2','lambda3','lambda4','meanVar','meanCov'])
Master_dim_stim_nonOS = pd.DataFrame(columns=['nTooVar','nROIs','PR','sumLambdas','lambda1','lambda2','lambda3','lambda4','meanVar','meanCov'])


# Loop over all datasets
for dd in range(len(numDatasets)):
    
    ##########################################################################
    # Load data
    ##########################################################################
    
    idxDataset = numDatasets[dd]
    
    dataDate = dataSpecs.iloc[idxDataset]['Date']
    dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
    dataDepth = dataSpecs.iloc[idxDataset]['Depth']
    pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
    dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
    
    if boolV1:
        filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
                dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    else:
        filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
            dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    
    fluoDF = pd.read_hdf(filepath,'fluo')
    charROI = pd.read_hdf(filepath,'charROI')
    charTrials = pd.read_hdf(filepath,'charTrials')
    positionROI = pd.read_hdf(filepath,'positionROI')
    distROI = pd.read_hdf(filepath,'distROI')
    fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
    fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
    fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()
    
    
    fluo = np.array(fluoDF)
    nROI_init = fluo.shape[1]
    
    # Set extreme dff0 values to nan
    tmpIdx = np.where((fluo>1000)|(fluo<-1000))[0]
    fluo[tmpIdx] = np.nan
    
    varROI = np.nanvar(fluo,axis=0)
    plt.figure()
    plt.hist(varROI)
    plt.title('Variability of ROIs (before discarding)')
    
    #idxTooVar = np.where(varROI>100)[0]
    idxTooVar = np.where(varROI>np.percentile(varROI, 80))[0]
    
    print('There are '+str(len(idxTooVar))+'/'+str(nROI_init)+' ROIs which are highly variable')
    fluo = np.delete(fluo,idxTooVar,axis=1)
    charROI = charROI.iloc[np.setdiff1d(np.arange(charROI.shape[0]),idxTooVar)].reset_index()
    
    varROI = np.nanvar(fluo,axis=0)
    plt.figure()
    plt.hist(varROI)
    plt.title('Variability of ROIs (after discarding)')
    
    plt.figure()
    plt.hist(varROI/np.nanmean(fluo,axis=0))
    plt.title('Fano factor of ROIs (after discarding)')
    
    # Select the ROIs and frame type we are interested in
    idxOS = np.where(np.array(charROI['OS']))[0]
    idxNonOS = np.where(np.array(charROI['OS']==False))[0]
    idxBlank = np.where(charTrials['FrameType']=='Blank')[0]
    #idxStimulus = np.where(charTrials['FrameType']=='Stimulus')[0] ### !!!
    idxStimulus = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==45))[0] ### !!!
    ### !!! len(idxStimulus) is not always = len(idxBlank)
    
    thisType='full' # 'full','shared','private'
    
    # OS only, blank frames
    thisFluo = fluo[:,idxOS][idxBlank,:]
    prBOS, covBOS, lambdaBOS = functions_analyze.compute_dimensionality(thisFluo,type=thisType)
    
    # OS only, stimulus frames
    thisFluo = fluo[:,idxOS][idxStimulus,:]
    prSOS, covSOS, lambdaSOS = functions_analyze.compute_dimensionality(thisFluo,type=thisType)
    
    # Non-OS only, blank frames
    thisFluo = fluo[:,idxNonOS][idxBlank,:]
    prBnonOS, covBnonOS, lambdaBnonOS = functions_analyze.compute_dimensionality(thisFluo,type=thisType)
    
    # Non-OS only, stimulus frames
    thisFluo = fluo[:,idxNonOS][idxStimulus,:]
    prSnonOS, covSnonOS, lambdaSnonOS = functions_analyze.compute_dimensionality(thisFluo,type=thisType)
    
    # Keep the results in a list
    # !!! 'nTooVar' is the number of ROI discarded, irrepsectively if they are OS or nonOS !!!
    Master_dim_blank_OS = Master_dim_blank_OS.append({'nTooVar' : len(idxTooVar), 'nROIs' : len(idxOS), \
                                                      'PR' : prBOS, 'sumLambdas' : np.sum(lambdaBOS), \
                                                      'lambda1' : lambdaBOS[0], 'lambda2' : lambdaBOS[1], \
                                                      'lambda3' : lambdaBOS[2], 'lambda4' : lambdaBOS[3], \
                                                      'meanVar' : np.round(np.mean(np.diag(covBOS)),1), \
                                                      'meanCov' : np.round(np.mean(covBOS[np.triu_indices(covBOS.shape[0],k=0)]),1)}, \
                                                     ignore_index = True)

    Master_dim_stim_OS = Master_dim_stim_OS.append({'nTooVar' : len(idxTooVar), 'nROIs' : len(idxOS), \
                                                    'PR' : prSOS, 'sumLambdas' : np.sum(lambdaSOS), \
                                                      'lambda1' : lambdaSOS[0], 'lambda2' : lambdaSOS[1], \
                                                      'lambda3' : lambdaSOS[2], 'lambda4' : lambdaSOS[3], \
                                                      'meanVar' : np.round(np.mean(np.diag(covSOS)),1), \
                                                      'meanCov' : np.round(np.mean(covSOS[np.triu_indices(covSOS.shape[0],k=0)]),1)}, \
                                                     ignore_index = True)
    Master_dim_blank_nonOS = Master_dim_blank_nonOS.append({'nTooVar' : len(idxTooVar), 'nROIs' : len(idxNonOS), \
                                                            'PR' : prBnonOS, 'sumLambdas' : np.sum(lambdaBnonOS), \
                                                      'lambda1' : lambdaBnonOS[0], 'lambda2' : lambdaBnonOS[1], \
                                                      'lambda3' : lambdaBnonOS[2], 'lambda4' : lambdaBnonOS[3], \
                                                      'meanVar' : np.round(np.mean(np.diag(covBnonOS)),1), \
                                                      'meanCov' : np.round(np.mean(covBnonOS[np.triu_indices(covBnonOS.shape[0],k=0)]),1)}, \
                                                     ignore_index = True)
    Master_dim_stim_nonOS = Master_dim_stim_nonOS.append({'nTooVar' : len(idxTooVar), 'nROIs' : len(idxNonOS), \
                                                          'PR' : prSnonOS, 'sumLambdas' : np.sum(lambdaSnonOS), \
                                                      'lambda1' : lambdaSnonOS[0], 'lambda2' : lambdaSnonOS[1], \
                                                      'lambda3' : lambdaSnonOS[2], 'lambda4' : lambdaSnonOS[3], \
                                                      'meanVar' : np.round(np.mean(np.diag(covSnonOS)),1), \
                                                      'meanCov' : np.round(np.mean(covSnonOS[np.triu_indices(covSnonOS.shape[0],k=0)]),1)}, \
                                                     ignore_index = True)

 
    
    
#%% Plotting only PR

if boolPR:
    
    x = np.arange(len(numDatasets))
    width = 0.2  # the width of the bars
    
    # OS only - PR
    tmp1 = Master_dim_blank_OS['PR']/Master_dim_blank_OS['nROIs']
    tmp2 = Master_dim_stim_OS['PR']/Master_dim_stim_OS['nROIs']
    fig, ax = plt.subplots()
    ax.bar(x - width/2, tmp1, width, label='Spont')
    ax.bar(x + width/2, tmp2, width, label='Evoked')
    ax.set_ylabel(thisType+' PR / N')
    ax.set_title(dataName+' - OS only')
    ax.legend()
    
    # Non-OS only - PR
    tmp1 = Master_dim_blank_nonOS['PR']/Master_dim_blank_nonOS['nROIs']
    tmp2 = Master_dim_stim_nonOS['PR']/Master_dim_stim_nonOS['nROIs']
    fig, ax = plt.subplots()
    ax.bar(x - width/2, tmp1, width, label='Spont')
    ax.bar(x + width/2, tmp2, width, label='Evoked')
    ax.set_ylabel(thisType+' PR / N')
    ax.set_title(dataName+' - non-OS only')
    ax.legend()
    
    # PR ratio (Evoked-Spontaneous)/Spontaneous
    ratioOS = (Master_dim_stim_OS['PR']-Master_dim_blank_OS['PR'])/Master_dim_blank_OS['PR']
    meanRatioOS = np.mean(ratioOS)
    semRatioOS = np.std(ratioOS)/np.sqrt(len(ratioOS))
    ratioNonOS = (Master_dim_stim_nonOS['PR']-Master_dim_blank_nonOS['PR'])/Master_dim_blank_nonOS['PR']
    meanRatioNonOS = np.mean(ratioNonOS)
    semRatioNonOS = np.std(ratioNonOS)/np.sqrt(len(ratioNonOS))
    
    plt.figure()
    plt.errorbar(0,meanRatioOS,yerr=semRatioOS)
    plt.errorbar(1,meanRatioNonOS,yerr=semRatioNonOS)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.xticks(ticks=[0, 1],labels=['OS','nonOS'])
    plt.ylabel('Mean +/- SEM')
    plt.title('(PR evoked - PR spont) / (PR spont)')



#%% More extensive plotting of PR, parcentage of variance explained, covariance, etc...

if boolPR:
    
    x = np.arange(len(numDatasets))  # the label locations
    width = 0.2  # the width of the bars
    
    # OS only - PR
    fig, ax = plt.subplots()
    ax.bar(x - width/2, Master_dim_blank_OS['PR'], width, label='Spont')
    ax.bar(x + width/2, Master_dim_stim_OS['PR'], width, label='Evoked')
    ax.set_ylabel(thisType+' PR')
    ax.set_title(dataType+' - OS only')
    ax.legend()
    
    # Non-OS only - PR
    fig, ax = plt.subplots()
    ax.bar(x - width/2, Master_dim_blank_nonOS['PR'], width, label='Spont')
    ax.bar(x + width/2, Master_dim_stim_nonOS['PR'], width, label='Evoked')
    ax.set_ylabel(thisType+' PR')
    ax.set_title(dataType+' - non-OS only')
    ax.legend()
    
    # PR ratio (Evoked-Spontaneous)/Spontaneous
    ratioOS = (Master_dim_stim_OS['PR']-Master_dim_blank_OS['PR'])/Master_dim_blank_OS['PR']
    np.mean(ratioOS)
    ratioNonOS = (Master_dim_stim_nonOS['PR']-Master_dim_blank_nonOS['PR'])/Master_dim_blank_nonOS['PR']
    np.mean(ratioNonOS)
    
    # OS only - sum of lambdas
    fig, ax = plt.subplots()
    ax.bar(x - width/2, Master_dim_blank_OS['sumLambdas'], width, label='Spont')
    ax.bar(x + width/2, Master_dim_stim_OS['sumLambdas'], width, label='Evoked')
    ax.set_ylabel('Sum lambdas')
    ax.set_title(dataType+' - OS only')
    ax.legend()
    
    # Non-OS only - sum of lambdas
    fig, ax = plt.subplots()
    ax.bar(x - width/2, Master_dim_blank_nonOS['sumLambdas'], width, label='Spont')
    ax.bar(x + width/2, Master_dim_stim_nonOS['sumLambdas'], width, label='Evoked')
    ax.set_ylabel('Sum lambdas')
    ax.set_title(dataType+' - non-OS only')
    ax.legend()
    
    # OS only - lambdas/sumLambdas
    for i in range(4):
        fig, ax = plt.subplots()
        ax.bar(x - width/2, Master_dim_blank_OS['lambda'+str(i+1)]/Master_dim_blank_OS['sumLambdas'], width, label='Spont')
        ax.bar(x + width/2, Master_dim_stim_OS['lambda'+str(i+1)]/Master_dim_stim_OS['sumLambdas'], width, label='Evoked')
        ax.set_ylabel('lambda'+str(i+1)+'/sumLambdas')
        ax.set_title(dataType+' - OS only')
        ax.legend()
    
    # Non-OS only - lambdas/sumLambdas
    for i in range(4):
        fig, ax = plt.subplots()
        ax.bar(x - width/2, Master_dim_blank_nonOS['lambda'+str(i+1)]/Master_dim_blank_nonOS['sumLambdas'], width, label='Spont')
        ax.bar(x + width/2, Master_dim_stim_nonOS['lambda'+str(i+1)]/Master_dim_stim_nonOS['sumLambdas'], width, label='Evoked')
        ax.set_ylabel('lambda'+str(i+1)+'/sumLambdas')
        ax.set_title(dataType+' - non-OS only')
        ax.legend()
    
    # OS only - mean variance
    fig, ax = plt.subplots()
    ax.bar(x - width/2, Master_dim_blank_OS['meanVar'], width, label='Spont')
    ax.bar(x + width/2, Master_dim_stim_OS['meanVar'], width, label='Evoked')
    ax.set_ylabel('Mean var')
    ax.set_title(dataType+' - OS only')
    ax.legend()
    
    # Non-OS only - mean variance
    fig, ax = plt.subplots()
    ax.bar(x - width/2, Master_dim_blank_nonOS['meanVar'], width, label='Spont')
    ax.bar(x + width/2, Master_dim_stim_nonOS['meanVar'], width, label='Evoked')
    ax.set_ylabel('Mean var')
    ax.set_title(dataType+' - non-OS only')
    ax.legend()
    
    # OS only - mean covariance
    fig, ax = plt.subplots()
    ax.bar(x - width/2, Master_dim_blank_OS['meanCov'], width, label='Spont')
    ax.bar(x + width/2, Master_dim_stim_OS['meanCov'], width, label='Evoked')
    ax.set_ylabel('Mean cov')
    ax.set_title(dataType+' - OS only')
    ax.legend()
    
    # Non-OS only - mean variance
    fig, ax = plt.subplots()
    ax.bar(x - width/2, Master_dim_blank_nonOS['meanCov'], width, label='Spont')
    ax.bar(x + width/2, Master_dim_stim_nonOS['meanCov'], width, label='Evoked')
    ax.set_ylabel('Mean cov')
    ax.set_title(dataType+' - non-OS only')
    ax.legend()



#%% Dimensionality of spontaneous versus evoked

# Loop over all datasets
for dd in range(len(numDatasets)):
    
    ##########################################################################
    # Load data
    ##########################################################################
    
    idxDataset = numDatasets[dd]
    
    dataDate = dataSpecs.iloc[idxDataset]['Date']
    dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
    dataDepth = dataSpecs.iloc[idxDataset]['Depth']
    pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
    dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
    
    if boolV1:
        filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
                dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    else:
        filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
            dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    
    fluoINIT = pd.read_hdf(filepath,'fluo')
    charROI = pd.read_hdf(filepath,'charROI')
    charTrials = pd.read_hdf(filepath,'charTrials')
    positionROI = pd.read_hdf(filepath,'positionROI')
    distROI = pd.read_hdf(filepath,'distROI')
    fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
    fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
    fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()
    
    fluo = np.array(fluoINIT)
    nROI_init = fluo.shape[1]
    
    varROI = np.var(fluoINIT,axis=0)
    idxTooVar = np.where(varROI>np.percentile(varROI, 50))[0]
    print('There are '+str(len(idxTooVar))+'/'+str(fluoINIT.shape[1])+' ROIs which are highly variable')
    fluo = np.delete(fluo,idxTooVar,axis=1)
    charROI = charROI.iloc[np.setdiff1d(np.arange(charROI.shape[0]),idxTooVar)].reset_index()

    
    # Select the ROIs and frame type we are interested in
    idxOS = np.where(np.array(charROI['OS']))[0]
    idxNonOS = np.where(np.array(charROI['OS']==False))[0]
    idxBlank = np.where(charTrials['FrameType']=='Blank')[0]
    #idxStimulus = np.where(charTrials['FrameType']=='Stimulus')[0] ### !!!
    idxStimulus = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==45))[0] ### !!!
    ### !!! len(idxStimulus) is not always = len(idxBlank)
    
    tmpNumROIs = np.arange(10,np.min((50+10,len(idxOS))),10)
    allPR_b,allPR_s = functions_analyze.compute_dimensionalityBootstrap_spontVSevoked(fluo[:,idxOS],idxBlank,idxStimulus,nROIs=tmpNumROIs,nTimes=100)
    
    # tmpNumROIs = np.arange(10,np.min((50+10,len(idxNonOS))),10)
    # allPR_b,allPR_s = functions_analyze.compute_dimensionalityBootstrap_spontVSevoked(fluo[:,idxNonOS],idxBlank,idxStimulus,nROIs=tmpNumROIs,nTimes=100)


#%% Pearson correlation between 1st motion PC and all ROIs

# Initialize dataframes where to save the Pearson corr info
Master_PearsonCorr = pd.DataFrame(columns=['spont','evoked','OS'])


# Loop over all datasets
for dd in range(len(numDatasets)):
    
    ##########################################################################
    # Load data
    ##########################################################################
    
    idxDataset = numDatasets[dd]
    
    dataDate = dataSpecs.iloc[idxDataset]['Date']
    dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
    dataDepth = dataSpecs.iloc[idxDataset]['Depth']
    pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
    dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
    
    if boolV1:
        filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
                dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist2d5um.hdf'
    else:
        filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
            dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '.hdf'
    
    fluo = pd.read_hdf(filepath,'fluo')
    charROI = pd.read_hdf(filepath,'charROI')
    charTrials = pd.read_hdf(filepath,'charTrials')
    positionROI = pd.read_hdf(filepath,'positionROI')
    distROI = pd.read_hdf(filepath,'distROI')
    fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
    fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
    fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()
    
    idxOS = np.where(np.array(charROI['OS']))[0]
    idxNonOS = np.where(np.array(charROI['OS']==False))[0]
    idxBlank = np.where(charTrials['FrameType']=='Blank')[0]
    idxStimulus = np.where(charTrials['FrameType']=='Stimulus')[0] ### !!!
    #idxStimulus = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==45))[0] ### !!!
    
    if len(idxBlank) < len(idxStimulus):
        idxStimulus = idxStimulus[0:len(idxBlank)]
    
    motSVD1_blank = charTrials['motSVD1'].iloc[idxBlank]
    motSVD1_stim = charTrials['motSVD1'].iloc[idxStimulus]
    # motSVD1_blank = charTrials['pupilArea'].iloc[idxBlank]
    # motSVD1_stim = charTrials['pupilArea'].iloc[idxStimulus]
    
    # Data for this dataset
    tmp_PearsonCorr = pd.DataFrame(columns=['spont','evoked','OS'])
    tmp_PearsonCorr['spont'] = fluo.iloc[idxBlank].corrwith(motSVD1_blank)
    tmp_PearsonCorr['evoked'] = fluo.iloc[idxStimulus].corrwith(motSVD1_stim)
    tmp_PearsonCorr['OS'] = charROI['OS']
    
    # Write the data into the dataframe
    Master_PearsonCorr = pd.concat([Master_PearsonCorr,tmp_PearsonCorr],axis=0,ignore_index=True)


# Plotting
thoseBins = np.linspace(-0.2,0.2,21) # np.linspace(-1,1,21)
fig, axes = plt.subplots(nrows=2, ncols=1)
Master_PearsonCorr.iloc[idxOS].plot.hist(alpha=0.5,title='Pearson corr - OS',ax=axes[0],bins=thoseBins)
Master_PearsonCorr.iloc[idxNonOS].plot.hist(alpha=0.5,title='Pearson corr - nonOS',ax=axes[1],bins=thoseBins)

Master_diffP = pd.concat([Master_PearsonCorr['evoked'].abs()-Master_PearsonCorr['spont'].abs(),Master_PearsonCorr['OS']],axis=1)
fig, axes = plt.subplots(nrows=2, ncols=1)
Master_diffP.iloc[idxOS].plot.hist(alpha=0.5,title='Diff abs. Pearson corr - OS',ax=axes[0],bins=thoseBins)
Master_diffP.iloc[idxNonOS].plot.hist(alpha=0.5,title='Diff abs. Pearson corr - nonOS',ax=axes[1],bins=thoseBins)
print('Diff abs. Pearson corr for OS (mu +/- SEM) is '+str(round(Master_diffP[0].iloc[idxOS].mean(),3))+' +/- '+str(round(Master_diffP[0].iloc[idxOS].sem(),3)))
print('Diff abs. Pearson corr for nonOS (mu +/- SEM) is '+str(round(Master_diffP[0].iloc[idxNonOS].mean(),3))+' +/- '+str(round(Master_diffP[0].iloc[idxNonOS].sem(),3)))

Master_diffP = pd.concat([Master_PearsonCorr['evoked']-Master_PearsonCorr['spont'],Master_PearsonCorr['OS']],axis=1)
fig, axes = plt.subplots(nrows=2, ncols=1)
Master_diffP.iloc[idxOS].plot.hist(alpha=0.5,title='Diff Pearson corr - OS',ax=axes[0],bins=thoseBins)
Master_diffP.iloc[idxNonOS].plot.hist(alpha=0.5,title='Diff Pearson corr - nonOS',ax=axes[1],bins=thoseBins)
print('Diff Pearson corr for OS (mu +/- SEM) is '+str(round(Master_diffP[0].iloc[idxOS].mean(),3))+' +/- '+str(round(Master_diffP[0].iloc[idxOS].sem(),3)))
print('Diff Pearson corr for nonOS (mu +/- SEM) is '+str(round(Master_diffP[0].iloc[idxNonOS].mean(),3))+' +/- '+str(round(Master_diffP[0].iloc[idxNonOS].sem(),3)))



#%% Pearson correlation between 1st motion PC and all ROIs - After having saved the results (to save computation time)

filepath = globalParams.processedDataDir + 'PearsonCorr_motSVD1_eachROI.hdf'

Master_PearsonCorr_L23 = pd.read_hdf(filepath,'Master_PearsonCorr_L23')
Master_PearsonCorr_L4 = pd.read_hdf(filepath,'Master_PearsonCorr_L4')
Master_PearsonCorr_boutonsL23 = pd.read_hdf(filepath,'Master_PearsonCorr_boutonsL23')


Master_PearsonCorr = Master_PearsonCorr_boutonsL23[['spont','evoked']]
Master_OS = Master_PearsonCorr_boutonsL23['OS']

# Parameters
thoseBins = np.linspace(-0.45,0.45,51)
numDigits = 4

# Plotting - OS and nonOS separately
fig, axes = plt.subplots(nrows=2, ncols=1)
Master_PearsonCorr.loc[Master_OS==True].plot.hist(alpha=0.5,title='OS - Pearson corr',ax=axes[0],bins=thoseBins)
Master_PearsonCorr.loc[Master_OS==False].plot.hist(alpha=0.5,title='Pearson corr - nonOS',ax=axes[1],bins=thoseBins)
print('Pearson corr for OS spont (mu +/- SEM) is '+str(round(Master_PearsonCorr['spont'].loc[Master_OS==True].mean(),numDigits))+' +/- '+str(round(Master_PearsonCorr['spont'].loc[Master_OS==True].sem(),numDigits)))
print('Pearson corr for OS evoked (mu +/- SEM) is '+str(round(Master_PearsonCorr['evoked'].loc[Master_OS==True].mean(),numDigits))+' +/- '+str(round(Master_PearsonCorr['evoked'].loc[Master_OS==True].sem(),numDigits)))
print('Pearson corr for nonOS spont (mu +/- SEM) is '+str(round(Master_PearsonCorr['spont'].loc[Master_OS==False].mean(),numDigits))+' +/- '+str(round(Master_PearsonCorr['spont'].loc[Master_OS==False].sem(),numDigits)))
print('Pearson corr for nonOS evoked (mu +/- SEM) is '+str(round(Master_PearsonCorr['evoked'].loc[Master_OS==False].mean(),numDigits))+' +/- '+str(round(Master_PearsonCorr['evoked'].loc[Master_OS==False].sem(),numDigits)))

Master_diffP = Master_PearsonCorr['evoked']-Master_PearsonCorr['spont']
fig, axes = plt.subplots(nrows=2, ncols=1)
Master_diffP.loc[Master_OS==True].plot.hist(alpha=0.5,title='Diff Pearson corr - OS',ax=axes[0],bins=thoseBins)
Master_diffP.loc[Master_OS==False].plot.hist(alpha=0.5,title='Diff Pearson corr - nonOS',ax=axes[1],bins=thoseBins)
print('Diff Pearson corr for OS (mu +/- SEM) is '+str(round(Master_diffP.loc[Master_OS==True].mean(),numDigits))+' +/- '+str(round(Master_diffP.loc[Master_OS==True].sem(),numDigits)))
print('Diff Pearson corr for nonOS (mu +/- SEM) is '+str(round(Master_diffP.loc[Master_OS==False].mean(),numDigits))+' +/- '+str(round(Master_diffP.loc[Master_OS==False].sem(),numDigits)))

# Plotting - OS and nonOS together
plt.figure()
Master_PearsonCorr.plot.hist(alpha=0.5,title='Pearson corr - all ROIs',bins=thoseBins)
print('Pearson corr for all ROIs spont (mu +/- SEM) is '+str(round(Master_PearsonCorr['spont'].mean(),numDigits))+' +/- '+str(round(Master_PearsonCorr['spont'].sem(),numDigits)))
print('Pearson corr for all ROIs evoked (mu +/- SEM) is '+str(round(Master_PearsonCorr['evoked'].mean(),numDigits))+' +/- '+str(round(Master_PearsonCorr['evoked'].sem(),numDigits)))

Master_diffP = Master_PearsonCorr['evoked']-Master_PearsonCorr['spont']
plt.figure()
Master_diffP.plot.hist(alpha=0.5,title='Diff Pearson corr - all ROIs',bins=thoseBins)
print('Diff Pearson corr for all ROIs (mu +/- SEM) is '+str(round(Master_diffP.mean(),numDigits))+' +/- '+str(round(Master_diffP.sem(),numDigits)))


#%% Pearson correlation between each pair of bouton to L2/3 and L2/3 ROI

# Parameters
dataType = 'L23_thalamicBoutons'
numDatasets = np.array((0,2,3,5,6))
dataNeuropilSub_L23 = globalParams.neuropilSub[3]
dataNeuropilSub_boutonsL23 = globalParams.neuropilSub[0]
filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')


# Initialize dataframes where to save the Pearson corr info
Master_PearsonCorr_spont = pd.DataFrame(columns=['spont','evoked','L23_OS','boutons_OS'])


# Loop over all datasets
for dd in range(len(numDatasets)):
    
    ##########################################################################
    # Load data
    ##########################################################################
    
    idxDataset = numDatasets[dd]
    
    dataDate = dataSpecs.iloc[idxDataset]['Date']
    dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
    dataDepth = dataSpecs.iloc[idxDataset]['Depth']
    pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
    dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
    
    ### L2/3 ROIs ###
    filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub_L23 + '_threshDist2d5um.hdf'
    
    fluo = pd.read_hdf(filepath,'fluo')
    charROI = pd.read_hdf(filepath,'charROI')
    idxOS = np.where(np.array(charROI['OS']))[0]
    idxNonOS = np.where(np.array(charROI['OS']==False))[0]
    
    charTrials = pd.read_hdf(filepath,'charTrials')
    idxBlank = np.where(charTrials['FrameType']=='Blank')[0]
    idxStimulus = np.where(charTrials['FrameType']=='Stimulus')[0] ### !!!
    #idxStimulus = np.where((charTrials['FrameType']=='Stimulus') & (charTrials['Orientation']==45))[0] ### !!!
    
    # if len(idxBlank) < len(idxStimulus):
    #     idxStimulus = idxStimulus[0:len(idxBlank)]
    
    
    ### Boutons to L2/3 ###
    filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub_boutonsL23 + '.hdf'
    
    fluo_boutons = pd.read_hdf(filepath,'fluo')
    charROI_boutons = pd.read_hdf(filepath,'charROI')
    idxOS_boutons = np.where(np.array(charROI['OS']))[0]
    idxNonOS_boutons = np.where(np.array(charROI['OS']==False))[0]
    
    ###
    zDF_spont = pd.concat([fluo_boutons.iloc[idxBlank],fluo.iloc[idxBlank]],axis=1)
    zDF_evoked = pd.concat([fluo_boutons.iloc[idxStimulus],fluo.iloc[idxStimulus]],axis=1)
    
    zDF = zDF_spont
    zPearsonCorr = np.array(zDF.corr(method='pearson'))
    nBoutons = fluo_boutons.shape[1]
    nL23 = fluo.shape[1]
    btwPearsonCorr = zPearsonCorr[:,nBoutons:nBoutons+nL23]
    btwPearsonCorr = btwPearsonCorr[0:nBoutons,:]
    print('Mean between corr spont: '+str(np.mean(btwPearsonCorr.flatten())))
    print('Std between corr spont: '+str(np.std(btwPearsonCorr.flatten())))
    
    withinL23 = zPearsonCorr[:,nBoutons:nBoutons+nL23]
    withinL23 = withinL23[nBoutons:nBoutons+nL23,:]
    idx = np.triu_indices(nL23, k=1) # indices of upper triangular matrix
    withinL23 = withinL23[idx]
    print('Mean within L2/3 corr spont: '+str(np.mean(withinL23)))
    print('Std withi L2/3 corr spont: '+str(np.std(withinL23))) 
    
    withinBL23 = zPearsonCorr[:,0:nBoutons]
    withinBL23 = withinBL23[0:nBoutons,:]
    idx = np.triu_indices(nBoutons, k=1) # indices of upper triangular matrix
    withinBL23 = withinBL23[idx]
    print('Mean within L2/3 corr spont: '+str(np.mean(withinL23)))
    print('Std withi L2/3 corr spont: '+str(np.std(withinL23)))
    
    zDF = zDF_evoked
    zPearsonCorr = np.array(zDF.corr(method='pearson'))
    nBoutons = fluo_boutons.shape[1]
    nL23 = fluo.shape[1]
    btwPearsonCorr = zPearsonCorr[:,nBoutons:nBoutons+nL23]
    btwPearsonCorr = btwPearsonCorr[0:nBoutons,:]
    print('Mean between corr evoked: '+str(np.mean(btwPearsonCorr.flatten())))
    print('Std between corr evoked: '+str(np.std(btwPearsonCorr.flatten())))
    
    withinL23 = zPearsonCorr[:,nBoutons:nBoutons+nL23]
    withinL23 = withinL23[nBoutons:nBoutons+nL23,:]
    idx = np.triu_indices(nL23, k=1) # indices of upper triangular matrix
    withinL23 = withinL23[idx]
    print('Mean within L2/3 corr evoked: '+str(np.mean(withinL23)))
    print('Std withi L2/3 corr evoked: '+str(np.std(withinL23))) 
    
    withinBL23 = zPearsonCorr[:,0:nBoutons]
    withinBL23 = withinBL23[0:nBoutons,:]
    idx = np.triu_indices(nBoutons, k=1) # indices of upper triangular matrix
    withinBL23 = withinBL23[idx]
    print('Mean within L2/3 corr evoked: '+str(np.mean(withinL23)))
    print('Std withi L2/3 corr evoked: '+str(np.std(withinL23))) 
    ###

    # Data for this dataset
    tmp_PearsonCorr = pd.DataFrame(columns=['spont','evoked','L23_OS','boutons_OS'])
    tmp_PearsonCorr['spont'] = fluo.iloc[idxBlank].corrwith(fluo_boutons.iloc[idxBlank],axis=0)
    tmp_PearsonCorr['evoked'] = fluo.iloc[idxStimulus].corrwith(motSVD1_stim)
    tmp_PearsonCorr['L23_OS'] = charROI['OS']
    tmp_PearsonCorr['boutons_OS'] = charROI_boutons['OS']
    
    # Write the data into the dataframe
    Master_PearsonCorr = pd.concat([Master_PearsonCorr,tmp_PearsonCorr],axis=0,ignore_index=True)    



