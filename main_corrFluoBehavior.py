# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:07:12 2022

@author: Olivia Gozel

Compute correlation of fluorescence activity with face motion and pupil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.linear_model as lm
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import signal


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze




#%% Parameters of the postprocessed data to analyze

### TO CHOOSE ###
dataType = 'L4_cytosolic' # 'L4_cytosolic' or 'L23_thalamicBoutons'
boolV1 = True # True or False
numDatasets = np.arange(11)
# np.arange(11) for V1 L4
# np.arange(8) for V1 L2/3
# np.array((0,2,3,5,6)) for L2/3 thalamic boutons
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor
# globalParams.neuropilSub[3] for V1 data
# globalParams.neuropilSub[0] for thalamic boutons

# Choose a type of analysis
analysisType = 'LR_motSVD_avgFluo'
# 'LR_motSVD_firstNcFluoPCs' : linear regression between all 500 motion PCs and the first nc+1 fluo PCs
# 'LR_motSVD_avgFluo' : linear regression between all 500 motion PCs and the average fluorescence

numCVF = 10 # number of cross-validation folds

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



#%% Functions to detrend the data

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


#%% Average fluorescence and coefficient of variation

Master_scoreTrain = []
Master_scoreTest = []

# Loop over all datasets
for dd in range(len(numDatasets)):
    
    tmp_scoreTrain = []
    tmp_scoreTest = []
    
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
    charROI = pd.read_hdf(filepath,'charROI')
    motSVD = pd.read_hdf(filepath,'motSVD')
    
    
    ##########################################################################
    # Use a regression model to explain neuronal variability
    ##########################################################################
    
    if motSVD is not None:
        
        #######################################################################
        # Linear regression between all 500 motion PCs and the first 1 to 1+nc fluo PCs
        #######################################################################
        if analysisType == 'LR_motSVD_firstNcFluoPCs':
        
            X = np.array(motSVD)
            Y = np.array(fluo) # All ROIs and all frames for now
            
            # We need to interpolate nan values, here we use the k-Nearest Neighbors approach (the feature of the neighbors are averaged uniformly)
            imputer = KNNImputer(n_neighbors=2, weights="uniform")
            Y = imputer.fit_transform(Y)
            
            # Scalar normalization of data
            sc = StandardScaler()
            Y = sc.fit_transform(Y)
            
            Master_scoreTrain = []
            Master_scoreTest = []
            
            for nc in range(200): # range(Y.shape[1])
            
                # Split data into training and testing sets
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
                
                # Build the model
                pca = PCA(n_components=1+nc) # PCA() to keep all components, otherwise PCA(n_components=3) for example
                Y_train = pca.fit_transform(Y_train)
                Y_test = pca.transform(Y_test)
                
                # # Compute the cumulative percentage of variance explained
                # explVarCumsum_Y = np.cumsum(pca.explained_variance_ratio_)
                # plt.figure()
                # plt.plot(100*explVarCumsum_Y)
                # plt.ylabel('Cumulative variance explained')
                # plt.xlabel('Principal component number')
                
                # Regression model to explain fluo by face motion PCs
                LRmodel = lm.LinearRegression() # lm.Ridge(alpha=0.5)
                LRmodel.fit(X_train, Y_train)
                # betaParams = LRmodel.coef_
                score_train = LRmodel.score(X_train, Y_train)
                score_test = LRmodel.score(X_test, Y_test)
                
                Master_scoreTrain.append(score_train)
                Master_scoreTest.append(score_test)
            
            plt.figure()
            plt.plot(Master_scoreTrain,label='LR score - train')
            plt.plot(Master_scoreTest,label='LR score - test')
            plt.xlabel('Number of fluo PCs')
            plt.legend()
        
        #######################################################################
        # Linear regression between all 500 motion PCs and the average fluo activity
        #######################################################################
        if analysisType == 'LR_motSVD_avgFluo':
            
            for cvf in range(numCVF):
                X = np.array(motSVD)
                if dd==10: # hack for now for L4_cytosolic
                    X = X[0:25000]
                
                # Select half of the ROIs
                numROI = fluo.shape[1]
                numROIhalf = np.floor(numROI/2).astype(int)
                idxSel = np.random.permutation(numROI)[0:numROIhalf]
                
                ### Start detrend the data
                for s in range(len(dataSessions)):
                    thisFluo = fluo.loc[charTrials['Session']==dataSessions[s]]
                    fluo_interp = fluo.interpolate()
                    col_withNan = np.where(fluo_interp.isna().any())[0]
                
                    # If there are still nan elements in the dataframe, most probably at the beginning
                    # or end of the timeseries, then replace then with mean value
                    if len(col_withNan) > 0: 
                        for c in range(len(col_withNan)):
                            tmpCol = col_withNan[c]
                            fluo_interp[tmpCol] = fluo_interp[tmpCol].where(~np.isnan(fluo_interp[tmpCol]),fluo_interp[tmpCol].mean())
                    
                    #thisFluo = pd.DataFrame(signal.detrend(fluo_interp))
                    thisFluo = pd.DataFrame(butter_highpass_filter(signal.detrend(fluo_interp), 0.1, dataFR))
                ### End detrend the data
                
                Y = np.array(thisFluo)[:,idxSel]
                Y = np.nanmean(Y,axis=1)
                
                # Y = np.nanmean(np.array(fluo),axis=1) # Average over all ROIs and all frames for now
                
                
                # Split data into training and testing sets
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) # if we want to use a seed: random_state=0
                
                # Regression model to explain average fluo by face motion PCs
                LRmodel = lm.LinearRegression() # lm.Ridge(alpha=0.5)
                LRmodel.fit(X_train, Y_train)
                score_train = LRmodel.score(X_train, Y_train)
                score_test = LRmodel.score(X_test, Y_test)
                
                tmp_scoreTrain.append(score_train)
                tmp_scoreTest.append(score_test)
            
            # Master_scoreTrain.append(score_train)
            # Master_scoreTest.append(score_test)
            if dd==0:
                Master_scoreTrain = tmp_scoreTrain
                Master_scoreTest = tmp_scoreTest
            else:
                Master_scoreTrain = np.vstack((Master_scoreTrain,tmp_scoreTrain))
                Master_scoreTest = np.vstack((Master_scoreTest,tmp_scoreTest))
            
            # # Plot at the end:
            # plt.figure()
            # plt.plot(Master_scoreTrain,label='LR score - train')
            # plt.plot(Master_scoreTest,label='LR score - test')
            # plt.xlabel('L4 datasets')
            # plt.legend()
            # plt.figure()
            # plt.errorbar(numDatasets,np.mean(Master_scoreTrain,axis=1),yerr=np.std(Master_scoreTrain,axis=1)/np.sqrt(numCVF),label='LR score - train')
            # plt.errorbar(numDatasets,np.mean(Master_scoreTest,axis=1),yerr=np.std(Master_scoreTest,axis=1)/np.sqrt(numCVF),label='LR score - test')
            # plt.xlabel(dataName+' datasets')
            # plt.legend()
            
        
        #######################################################################
        # Some old code
        #######################################################################
        if analysisType == 'old':
            
            Master_LRscore = []
            
            # Check linear ridge regression score
            for roi in range(fluo.shape[1]):
                X = np.array(motSVD)
                y = np.array(fluo[roi])
                idxNan = np.where(np.isnan(y))[0]
                y = np.delete(y,idxNan)
                X = np.delete(X,idxNan,axis=0)
                model.fit(X, y)
                # betaParams = model.coef_
                thisScore = model.score(X, y)
                Master_LRscore.append(thisScore)
            
            bins = np.arange(0,0.6,0.025) # np.arange(0,1,0.025)
            
            # Plot histogram of linear regression score        
            tmpNotVE = np.where(charROI['VisuallyEvoked']==False)[0]
            tmpVE = np.where(charROI['VisuallyEvoked']==True)[0]
            tmpOS = np.where(charROI['OS']==True)[0]
            plt.figure()
            plt.hist(np.array(Master_LRscore)[tmpNotVE],bins,alpha=0.5,label='Not VE')
            plt.hist(np.array(Master_LRscore)[tmpVE],bins,alpha=0.5,label='VE')
            plt.hist(np.array(Master_LRscore)[tmpOS],bins,alpha=0.5,label='OS')
            plt.xlabel('LR score')
            plt.legend()
            
            # Linear regression of the first eigenvectors of the fluorescence activity using motSVD
            Y,S,Vh = np.linalg.svd(fluo.cov())
            Master_LRscore_fluoPC = []
            for thisPC in range(Y.shape[1]):
                X = np.array(motSVD)
                y = np.dot(np.array(fluo),Y[:,thisPC])/np.linalg.norm(Y[:,thisPC])
                idxNan = np.where(np.isnan(y))[0]
                y = np.delete(y,idxNan)
                X = np.delete(X,idxNan,axis=0)
                model.fit(X, y)
                thisScore = model.score(X, y)
                Master_LRscore_fluoPC.append(thisScore)
            
            plt.figure()
            plt.plot(Master_LRscore_fluoPC,label='LRscore')
            plt.plot((np.cumsum(S)/np.sum(S))*np.max(Master_LRscore_fluoPC),label='Normalized cumsum var. expl.')
            plt.xlabel('Fluo PCs')
            plt.legend()
        
    
    
    
    
    
    
    
    