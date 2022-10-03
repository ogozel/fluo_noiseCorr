# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 12:36:19 2022

Plot some figures for Na's grant renewal

@author: Olivia Gozel
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
    
    
#%% Plot image of covariance matrix of boutons
# all boutons
# only stimulus frames

idxSelFrames = np.where(charTrials['FrameType']=='Stimulus')[0]
thisFluo = np.array(fluo)[idxSelFrames,:]
thisCharTrials = charTrials.loc[idxSelFrames].reset_index(drop=True)

# Select trials of a given orientation
o = 0
theseFrames = np.where(thisCharTrials['Orientation']==globalParams.ori[o])[0]
thisFluo_perOri = thisFluo[theseFrames]
thisFluo_perOri_df = pd.DataFrame(thisFluo_perOri)

# Full neural covariance matrix
cov_full = thisFluo_perOri_df.cov()

# Full neural correlation matrix
corr_full = thisFluo_perOri_df.corr()
zCorr = np.array(corr_full)





#%% Top 4 eigenvalues of the covariance matrix as a function of the number of trials

charTrials_stim = thisCharTrials.loc[theseFrames].reset_index(drop=True)

trialNumbers = np.unique(charTrials_stim['Trial'])
numTotTrials = len(trialNumbers)

Master_numTrials = np.arange(25,275,25)
numDraws = 10
numBoutons = thisFluo.shape[1]
Master_eigVal = np.zeros((numBoutons,numDraws,len(Master_numTrials)))

for nt in range(len(Master_numTrials)):
    for d in range(numDraws):
        
        # Select trials
        tmpNumTrials = Master_numTrials[nt]
        tmpTrials = np.random.permutation(numTotTrials)[:tmpNumTrials]
        tmpTrialNumbers = trialNumbers[tmpTrials]
        
        # Compute covariance matrix
        tmpFrames = np.where(np.isin(charTrials_stim['Trial'],tmpTrialNumbers))[0]
        tmpFluo = thisFluo_perOri[tmpFrames]
        tmpFluo_df = pd.DataFrame(tmpFluo)
        tmp_cov_full = tmpFluo_df.cov()
        
        # Sort eigenvalues from high to low
        eigVal = np.sort(np.linalg.eig(np.array(tmp_cov_full))[0])[::-1]
        
        # Write in big matrix
        Master_eigVal[:,d,nt] = eigVal
        
        
# Plot the first eigenvalues as a function of number of trials
plt.figure()
for i in range(4):
    thisVal = Master_eigVal[i,:,:]
    meanEigVal = np.mean(thisVal,axis=0)
    semEigVal = np.std(thisVal,axis=0)/np.sqrt(numDraws)
    
    plt.errorbar(Master_numTrials,meanEigVal,yerr=semEigVal,label='eigVal'+str(i+1))
    
plt.xlabel('Number of trials')
plt.ylabel('Eigenvalue')
plt.legend()
        
        



