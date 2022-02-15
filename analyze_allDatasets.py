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
boolV1 = False
numDatasets = np.array((0,2,3))
# np.arange(10) for V1 L4
# np.arange(4) for V1 L2/3
# np.array((0,2,3)) for L2/3 thalamic boutons
dataNeuropilSub = globalParams.neuropilSub[0] # choose a neuropil factor
# globalParams.neuropilSub[3] for V1 data
# globalParams.neuropilSub[0] for thalamic boutons

boolPR = True


filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')


# Dimensionality stuff
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
                dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist2d5um.hdf'
    else:
        filepath = globalParams.processedDataDir + dataType +'_boutons_' + dataDate + '_' + \
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
    
    
    ##########################################################################
    # Compute shared Participation Ratio
    ##########################################################################
    if boolPR:
        
        fluo = np.array(fluoINIT)
        varROI = np.var(fluoINIT,axis=0)
        plt.figure()
        plt.hist(varROI)
        plt.title('Variability of ROIs (before discarding)')
        idxTooVar = np.where(varROI>np.median(varROI,axis=0) + 3*np.std(varROI,axis=0))[0]
        if (dataType=='L4_cytosolic') & (idxDataset==8) & (boolV1==True): # HACK because a few very variable ROIs (order 1e7 to 1e35)
            idxTooVar = np.where(varROI>5e4)[0]
        if (dataType=='L23_thalamicBoutons') & (idxDataset==2) & (boolV1==True): # HACK because a few very variable ROIs (order 1e7 to 1e35)
            idxTooVar = np.where(varROI>1e5)[0]
        if (dataType=='L23_thalamicBoutons') & (idxDataset==3) & (boolV1==False): # HACK because a few very variable ROIs (order 1e7 to 1e35)
            idxTooVar = np.where(varROI>1e8)[0]
        print('There are '+str(len(idxTooVar))+'/'+str(fluoINIT.shape[1])+' ROIs which are highly variable')
        fluo = np.delete(fluo,idxTooVar,axis=1)
        charROI = charROI.iloc[np.setdiff1d(np.arange(charROI.shape[0]),idxTooVar)].reset_index()
        
        varROI = np.var(fluo,axis=0)
        plt.figure()
        plt.hist(varROI)
        plt.title('Variability of ROIs (after discarding)')
        
        # Select the ROIs and frame type we are interested in
        idxOS = np.where(np.array(charROI['OS']))[0]
        idxNonOS = np.where(np.array(charROI['OS']==False))[0]
        idxBlank = np.where(charTrials['FrameType']=='Blank')[0]
        #idxStimulus = np.where(charTrials['FrameType']=='Stimulus')[0]
        idxStimulus = np.where(charTrials['TrialFrame'].isin(20+np.arange(5)))[0]
        
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
        Master_dim_blank_OS = Master_dim_blank_OS.append({'nTooVar' : len(idxTooVar), 'nROIs' : fluo.shape[1], \
                                                          'PR' : prBOS, 'sumLambdas' : np.sum(lambdaBOS), \
                                                          'lambda1' : lambdaBOS[0], 'lambda2' : lambdaBOS[1], \
                                                          'lambda3' : lambdaBOS[2], 'lambda4' : lambdaBOS[3], \
                                                          'meanVar' : np.round(np.mean(np.diag(covBOS)),1), \
                                                          'meanCov' : np.round(np.mean(covBOS[np.triu_indices(covBOS.shape[0],k=0)]),1)}, \
                                                         ignore_index = True)
    
        Master_dim_stim_OS = Master_dim_stim_OS.append({'nTooVar' : len(idxTooVar), 'nROIs' : fluo.shape[1], \
                                                        'PR' : prSOS, 'sumLambdas' : np.sum(lambdaSOS), \
                                                          'lambda1' : lambdaSOS[0], 'lambda2' : lambdaSOS[1], \
                                                          'lambda3' : lambdaSOS[2], 'lambda4' : lambdaSOS[3], \
                                                          'meanVar' : np.round(np.mean(np.diag(covSOS)),1), \
                                                          'meanCov' : np.round(np.mean(covSOS[np.triu_indices(covSOS.shape[0],k=0)]),1)}, \
                                                         ignore_index = True)
        Master_dim_blank_nonOS = Master_dim_blank_nonOS.append({'nTooVar' : len(idxTooVar), 'nROIs' : fluo.shape[1], \
                                                                'PR' : prBnonOS, 'sumLambdas' : np.sum(lambdaBnonOS), \
                                                          'lambda1' : lambdaBnonOS[0], 'lambda2' : lambdaBnonOS[1], \
                                                          'lambda3' : lambdaBnonOS[2], 'lambda4' : lambdaBnonOS[3], \
                                                          'meanVar' : np.round(np.mean(np.diag(covBnonOS)),1), \
                                                          'meanCov' : np.round(np.mean(covBnonOS[np.triu_indices(covBnonOS.shape[0],k=0)]),1)}, \
                                                         ignore_index = True)
        Master_dim_stim_nonOS = Master_dim_stim_nonOS.append({'nTooVar' : len(idxTooVar), 'nROIs' : fluo.shape[1], \
                                                              'PR' : prSnonOS, 'sumLambdas' : np.sum(lambdaSnonOS), \
                                                          'lambda1' : lambdaSnonOS[0], 'lambda2' : lambdaSnonOS[1], \
                                                          'lambda3' : lambdaSnonOS[2], 'lambda4' : lambdaSnonOS[3], \
                                                          'meanVar' : np.round(np.mean(np.diag(covSnonOS)),1), \
                                                          'meanCov' : np.round(np.mean(covSnonOS[np.triu_indices(covSnonOS.shape[0],k=0)]),1)}, \
                                                         ignore_index = True)

 
    
    
#%% Plotting

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


