# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:58:45 2021

@author: Olivia Gozel

Functions to do the analyses
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import statsmodels.tsa.stattools as smt
import random as rnd
from sklearn.decomposition import FactorAnalysis

import globalParams



def compute_neuralPCs(fluo,bool_plot=0,title=None):
    
    # Mean-subtract the neural activity
    fluo_meanSub = np.array(fluo - np.mean(fluo,axis=0))

    # Full neural covariance matrix
    #c_full = np.matmul(np.transpose(fluo_meanSub), fluo_meanSub)

    # Full PCA (get all the PCs)
    pca = PCA()
    pca.fit(fluo_meanSub)

    # Principal components (rows of pca.components_) = eigenvectors of c_full
    #u = np.transpose(pca.components_)

    # Singular values squared = eigenvalues of c_full
    #eigenvals = np.square(pca.singular_values_)

    # Percentage of variance explained
    percVarExpl = pca.explained_variance_ratio_

    # Projection on the PCs
    #neuralProj = pca.fit_transform(fluo_meanSub)
    neuralProj = pca.fit_transform(fluo)
    
    # Plot the percentage of variance explained by the first latent variables
    if bool_plot:
        plt.figure()
        plt.plot(np.linspace(1,percVarExpl.shape[0],percVarExpl.shape[0]),100*percVarExpl)
        plt.xlabel('Eigenmode')
        plt.ylabel('Percentage of variance explained [%]')
        plt.xlim(0.9,5.1)
        plt.title(title)
    
    return neuralProj, percVarExpl



def plot_ftrials_neuralPCs(fluo,nFramesPerTrial,title=None,nInstances=10):
    
    nFrames = fluo.shape[0]
    nROIs = fluo.shape[1]
    
    nTrials = int(nFrames/nFramesPerTrial)
    
    # Vectors with number of randomly selected trials
    #numTrialsSel = np.arange(75,nTrials,50)
    numTrialsSel = np.round(np.linspace(50,nTrials,5)).astype(int)
    
    # Include the total number of trials
    if ~np.isin(nTrials,numTrialsSel):
        numTrialsSel = np.append(numTrialsSel,nTrials)
    
    
    percVarExpl = np.zeros((nROIs,len(numTrialsSel),nInstances))
    firstPC = np.zeros((nROIs,len(numTrialsSel),nInstances))
    percVarExpl[:] = np.nan
    firstPC[:] = np.nan
    
    # Loop over the number of trials randomly selected
    for n in range(0,len(numTrialsSel)):
        
        # Number of trials selected
        thisNumTrialsSel = numTrialsSel[n]
        
        # Loop over all the draw instances
        for draw in range(0,nInstances):
        
            # Randomly select some trials
            theseTrials = rnd.sample(range(0,nTrials),thisNumTrialsSel)
            tmp = [i * nFramesPerTrial for i in theseTrials]
            tmp = [range(i,i+nFramesPerTrial) for i in tmp]
            tmp = np.reshape(np.array(tmp),thisNumTrialsSel*nFramesPerTrial)
            thisFluo = fluo[tmp,:]
            
            # Mean-subtract the neural activity
            fluo_meanSub = np.array(thisFluo - np.mean(thisFluo,axis=0))
        
            # Full PCA (get all the PCs)
            pca = PCA()
            pca.fit(fluo_meanSub)
        
            # Principal components (rows of pca.components_) = eigenvectors of c_full
            # L2-normalized such that they have a norm of 1
            u = normalize(np.transpose(pca.components_),axis=0)
            
            # First principal component (normalized)
            u1 = u[:,0]
            if np.sum(u1)<0:
                u1 = -u1
            
            # Save first PC
            firstPC[:,n,draw] = u1
        
            # Singular values squared = eigenvalues of c_full
            #s = np.square(pca.singular_values_)
        
            # Percentage of variance explained
            percVarExpl[:,n,draw] = pca.explained_variance_ratio_
            
            # With all the data, one "draw" is sufficient
            if thisNumTrialsSel==nTrials:
                break
                
    
    
    # Plot the percentage of variance explained by the first latent variables
    # (contribution)
    plt.figure()
    for n in range(0,len(numTrialsSel)):
        plt.errorbar(range(1,fluo.shape[1]+1), \
                     100*np.nanmean(percVarExpl[:,n,:],axis=1), \
                         yerr=100*np.nanstd(percVarExpl[:,n,:],axis=1)/np.sqrt(nInstances), \
                             label=str(numTrialsSel[n])+' trials')
    plt.xlabel('Eigenmode')
    plt.ylabel('Percentage of variance explained [%]')
    plt.xlim(0.9,5.1)
    #plt.ylim(4,20)
    plt.legend()
    plt.title(title+' - nInstances='+str(nInstances))
    #plt.savefig('percVarExpl_fnTrials.eps', format='eps')
    
    # Plot the angle between the first PC using all the data and the first PC using only part of the data
    # (structure)
    u1_allTrials = firstPC[:,len(numTrialsSel)-1,0]
    meanAngle = np.empty(len(numTrialsSel))
    semAngle = np.empty(len(numTrialsSel))
    meanAngle[:] = np.nan
    semAngle[:] = np.nan
    for n in range(0,len(numTrialsSel)):
        thisAngle = np.empty(nInstances)
        thisAngle[:] = np.nan
        for draw in range(0,nInstances):
            if n<len(numTrialsSel)-1:
                # First PC using numTrialsSel[n] trials for each instance
                thisU1 = firstPC[:,n,draw]
                # Angle in degrees (nb: both vectors are already normalized, but there is some estimation error)
                thisAngle[draw] = (180/np.pi)*np.arccos(np.dot(thisU1,u1_allTrials)/(np.linalg.norm(thisU1)*np.linalg.norm(u1_allTrials)))
            else:
            #if n==len(numTrialsSel)-1:
                meanAngle[n] = 0
                semAngle[n] = 0
                break
        if n<len(numTrialsSel)-1:
            meanAngle[n] = np.nanmean(thisAngle)
            semAngle[n] = np.nanstd(thisAngle)/np.sqrt(nInstances)
    
    plt.figure()
    plt.errorbar(numTrialsSel, meanAngle, yerr=semAngle)
    plt.xlabel('Number of trials')
    plt.ylabel('Angle between PC1 and PC1 using all data [°]')
    plt.title(title+' - nInstances='+str(nInstances))
    #plt.ylim(0,30)
    #plt.savefig('angle_fnTrials.eps', format='eps')
    
    
    
def plot_traces(traceNeural,tracePupil=None,traceMotion=None,bool_pupil=0,bool_motion=0,allDepths=None): 
    
    if allDepths is None:
        
        fig, axs = plt.subplots(3)
        fig.suptitle('Time series')
        axs[0].plot(traceNeural)
        axs[0].set_title('neuralPC1')
        if bool_pupil:
            axs[1].plot(tracePupil)
            axs[1].set_title('pupilArea')
        if bool_motion:
            axs[2].plot(traceMotion)
            axs[2].set_title('motionPC1')
            
    else:
        
        for d in range(0,len(allDepths)):
            
            plt.figure()
            fig, axs = plt.subplots(3)
            fig.suptitle('Time series - depth = '+str(allDepths[d])+' nm')
            axs[0].plot(traceNeural[:,d])
            axs[0].set_title('neuralPC1')
            if bool_pupil:
                axs[1].plot(tracePupil)
                axs[1].set_title('pupilArea')
            if bool_motion:
                axs[2].plot(traceMotion)
                axs[2].set_title('motionPC1')
            plt.show()



def plot_autocorrelations(traceNeural,tracePupil=None,traceMotion=None,bool_pupil=0,bool_motion=0,allDepths=None):

    # neural
    if allDepths is None:

        ccf_output = smt.ccf(traceNeural, traceNeural, adjusted=False)
        lags = np.linspace(0,len(ccf_output)-1,len(ccf_output))
        idx = np.where(lags<=100)
        plt.figure()
        plt.stem(lags[idx],ccf_output[idx])
        plt.xlabel('lag')
        plt.ylabel('Auto-corr')
        plt.title('neuralPC1')
        
    else:
        
        for d in range(0,len(allDepths)):
            
            ccf_output = smt.ccf(traceNeural[:,d], traceNeural[:,d], adjusted=False)
            lags = np.linspace(0,len(ccf_output)-1,len(ccf_output))
            idx = np.where(lags<=100)
            plt.figure()
            plt.stem(lags[idx],ccf_output[idx])
            plt.xlabel('lag')
            plt.ylabel('Auto-corr')
            plt.title('neuralPC1 - depth = '+str(allDepths[d])+' nm')
            

    # pupil
    if bool_pupil:
        ccf_output = smt.ccf(tracePupil, tracePupil, adjusted=False)
        lags = np.linspace(0,len(ccf_output)-1,len(ccf_output))
        idx = np.where(lags<=100)
        plt.figure()
        plt.stem(lags[idx],ccf_output[idx])
        plt.xlabel('lag')
        plt.ylabel('Auto-corr')
        plt.title('pupilArea')

    # motion
    if bool_motion:
        ccf_output = smt.ccf(traceMotion, traceMotion, adjusted=False)
        lags = np.linspace(0,len(ccf_output)-1,len(ccf_output))
        idx = np.where(lags<=100)
        plt.figure()
        plt.stem(lags[idx],ccf_output[idx])
        plt.xlabel('lag')
        plt.ylabel('Auto-corr')
        plt.title('motionPC1')


# Plot cross-correlation between neural activity and behavior (pupil, face motion)
def plot_crosscorrelations(traceNeural,tracePupil=None,traceMotion=None,bool_pupil=0,bool_motion=0,allDepths=None):
    
    if allDepths is None:
    
        # neuralPC1 - pupilArea
        if bool_pupil:
            backwards = smt.ccf(tracePupil, traceNeural, adjusted=False)[::-1]
            forwards = smt.ccf(traceNeural, tracePupil, adjusted=False)
            ccf_output = np.r_[backwards[:-1], forwards]
            # flip cross-corr is the value at zero lag is negative
            if forwards[0]<0:
                ccf_output = -ccf_output
            lags = np.linspace(-len(backwards),len(forwards),2*len(forwards)-1)
            idx = np.where((lags>=-100) & (lags<=100))
            plt.figure()
            plt.stem(lags[idx],ccf_output[idx])
            plt.xlabel('lag')
            plt.ylabel('Cross-corr')
            plt.title('neuralPC1 - pupilArea')
        
        # neuralPC1 - motionPC1
        if bool_motion:
            backwards = smt.ccf(traceMotion, traceNeural, adjusted=False)[::-1]
            forwards = smt.ccf(traceNeural, traceMotion, adjusted=False)
            ccf_output = np.r_[backwards[:-1], forwards]
            # flip cross-corr is the value at zero lag is negative
            if forwards[0]<0:
                ccf_output = -ccf_output
            lags = np.linspace(-len(backwards),len(forwards),2*len(forwards)-1)
            idx = np.where((lags>=-100) & (lags<=100))
            plt.figure()
            plt.stem(lags[idx],ccf_output[idx])
            plt.xlabel('lag')
            plt.ylabel('Cross-corr')
            plt.title('neuralPC1 - motionPC1')
            
    else:
        
        for d in range(0,len(allDepths)):
            
            # neuralPC1 - pupilArea
            if bool_pupil:
                backwards = smt.ccf(tracePupil, traceNeural[:,d], adjusted=False)[::-1]
                forwards = smt.ccf(traceNeural[:,d], tracePupil, adjusted=False)
                ccf_output = np.r_[backwards[:-1], forwards]
                # flip cross-corr is the value at zero lag is negative
                if forwards[0]<0:
                    ccf_output = -ccf_output
                lags = np.linspace(-len(backwards),len(forwards),2*len(forwards)-1)
                idx = np.where((lags>=-100) & (lags<=100))
                plt.figure()
                plt.stem(lags[idx],ccf_output[idx])
                plt.xlabel('lag')
                plt.ylabel('Cross-corr')
                plt.title('neuralPC1 (depth='+str(allDepths[d])+'nm) - pupilArea')
            
            # neuralPC1 - motionPC1
            if bool_motion:
                backwards = smt.ccf(traceMotion, traceNeural[:,d], adjusted=False)[::-1]
                forwards = smt.ccf(traceNeural[:,d], traceMotion, adjusted=False)
                ccf_output = np.r_[backwards[:-1], forwards]
                # flip cross-corr is the value at zero lag is negative
                if forwards[0]<0:
                    ccf_output = -ccf_output
                lags = np.linspace(-len(backwards),len(forwards),2*len(forwards)-1)
                idx = np.where((lags>=-100) & (lags<=100))
                plt.figure()
                plt.stem(lags[idx],ccf_output[idx])
                plt.xlabel('lag')
                plt.ylabel('Cross-corr')
                plt.title('neuralPC1 (depth='+str(allDepths[d])+'nm) - motionPC1')
            
        
    
    # pupilArea - motionPC1
    if bool_pupil & bool_motion:
        backwards = smt.ccf(traceMotion, tracePupil, adjusted=False)[::-1]
        forwards = smt.ccf(tracePupil, traceMotion, adjusted=False)
        ccf_output = np.r_[backwards[:-1], forwards]
        # flip cross-corr is the value at zero lag is negative
        if forwards[0]<0:
            ccf_output = -ccf_output
        lags = np.linspace(-len(backwards),len(forwards),2*len(forwards)-1)
        idx = np.where((lags>=-100) & (lags<=100))
        plt.figure()
        plt.stem(lags[idx],ccf_output[idx])
        plt.xlabel('lag')
        plt.ylabel('Cross-corr')
        plt.title('pupilArea - motionPC1')


# Plot cross-correlation between V1 activity and the activity of thalamic boutons
def plot_crosscorrelations_ROIvsBoutons(fluoROI,fluoBouton,idxAssignedROI,idxOverlappingBoutons):
    
    fluoROI = np.array(fluoROI)
    fluoBouton = np.array(fluoBouton)
    
    # # V1 ROIs which have boutons
    # theseROI = np.unique(idxAssignedROI)
    
    # # Pick one V1 ROI
    # tmpROI = theseROI[1]
    # tmpFluoROI = fluoROI[:,tmpROI]
    
    # # Pick the assigned boutons
    # tmpBoutons = np.where(idxAssignedROI==tmpROI)[0]
    # nTmpBoutons = tmpBoutons.size
    
    # Pick one bouton
    tmp = 0
    tmpBouton = idxOverlappingBoutons[tmp]
    tmpFluoBouton = fluoBouton[:,tmpBouton]
    
    # Pick the assigned ROI
    tmpROI = idxAssignedROI[tmp+1] # !!!!!!!!!!!! dirty trick
    tmpFluoROI = fluoROI[:,tmpROI]
    
    # for i in range(0,nTmpBoutons):
    #     tmpFluoBouton = fluoBouton[:,tmpBoutons[i]]
        
    # Compute cross-correlation
    backwards = smt.ccf(tmpFluoBouton, tmpFluoROI, adjusted=False)[::-1]
    forwards = smt.ccf(tmpFluoROI, tmpFluoBouton, adjusted=False)
    ccf_output = np.r_[backwards[:-1], forwards]
    # flip cross-corr is the value at zero lag is negative
    if forwards[0]<0:
        ccf_output = -ccf_output
    lags = np.linspace(-len(backwards),len(forwards),2*len(forwards)-1)
    idx = np.where((lags>=-50) & (lags<=50))
    plt.figure()
    plt.stem(lags[idx],ccf_output[idx])
    plt.xlabel('lag')
    plt.ylabel('Cross-corr')
    plt.title('V1 fluo ('+str(tmpROI)+') - Thalamic bouton ('+str(tmpBouton)+') fluo')



# Plot pairwise correlation as a function of pairwise distance
# binSize in [um]
def plot_corr_fdist(fluo,distROI,startBin=2.5,binSize=10,title=None):
    
    # Mean-subtract the neural activity
    fluo_meanSub = np.array(fluo - np.mean(fluo,axis=0))
    
    # Full neural covariance matrix
    corr_full = np.corrcoef(fluo_meanSub, rowvar=False)
    
    # Pairwise correlations
    pairCorr = np.triu(corr_full,k=1).flatten()
    
    # Pairwise distances
    pairDist = np.triu(distROI,k=1).flatten()
    
    # Select only the non-zero pairwise correlations
    tmpIdx = np.where(pairCorr!=0)
    pairDist = pairDist[tmpIdx]
    pairCorr = pairCorr[tmpIdx]
    
    # Compute histogram
    numBin = int(700/binSize) # int(500/binSize)
    edges = np.linspace(startBin,700+startBin,numBin+1) #np.linspace(2.5,702.5,numBin+1) # np.linspace(2.5,502.5,numBin+1)
    sortedPairCorr = []
    meanBinPairCorr = np.zeros(numBin)
    semBinPairCorr = np.zeros(numBin)
    for i in range(0,numBin):
        startEdge = edges[i]
        endEdge = edges[i+1]
        thisIdx = np.where((pairDist>startEdge) & (pairDist<=endEdge))
        sortedPairCorr.append(pairCorr[thisIdx])
        meanBinPairCorr[i] = np.mean(pairCorr[thisIdx])
        semBinPairCorr[i] = np.std(pairCorr[thisIdx])/np.sqrt(np.array(thisIdx).shape[1])
    
    binCenters = np.linspace(startBin + binSize/2, startBin + numBin*binSize-binSize/2,numBin)
    # Plot errorbars
    plt.figure()
    plt.errorbar(binCenters, meanBinPairCorr, yerr=semBinPairCorr)
    plt.xlabel('Distance [um]')
    plt.ylabel('Correlation')
    plt.xlim(0,250)
    plt.ylim(0,0.5)
    plt.grid(axis = 'y')
    plt.title(title)
    #plt.savefig('corr_fdist.eps', format='eps')
    
    return binCenters, sortedPairCorr


# Plot noise pairwise correlations as a function of pairwise distance
# binSize in [um]
def plot_noiseCorr_fdist(fluo, distROI, charROI, charTrials, 
                         boolVisuallyEvoked, boolOS, boolIdenticalTuning, 
                         boolStimulus, startBin=5, binSize=5):
    
    '''Plot noise correlation as a function of pairwise distance'''
    
    if boolVisuallyEvoked:
        if boolOS:
            # OS ROIs are visually evoked
            idxSelROI = np.where(charROI['OS']==True)[0]
        else:
            idxSelROI = np.where((charROI['VisuallyEvoked']==True)
                                 &(charROI['OS']==False))[0]
    else:
        idxSelROI = np.where(charROI['VisuallyEvoked']==False)[0]
        
    if boolStimulus:
        idxSelFrames = np.where(charTrials['FrameType']=='Stimulus')[0]
    else:
        idxSelFrames = np.where(charTrials['FrameType']=='Blank')[0]
        
    # Extract the data
    thisDistROI = np.array(distROI)[idxSelROI,:][:,idxSelROI]
    thisFluo = np.array(fluo)[idxSelFrames,:][:,idxSelROI]
    thisCharTrials = charTrials.loc[idxSelFrames]
    thisCharROI = charROI.loc[idxSelROI]
    
    # Pairwise distances
    if not boolIdenticalTuning:
        pairDist = np.triu(thisDistROI, k=1).flatten()
    
    # For the histogram
    numBin = int(500/binSize)
    edges = np.linspace(startBin,500+startBin,numBin+1)
    binCenters = np.linspace(startBin + binSize/2, 
                             startBin + numBin*binSize-binSize/2, numBin)
    
    # Initialize list to write the pairwise correlations
    sortedPairCorr = []
    
    # Noise correlations, so take trials of each orientation separately
    for o in range(globalParams.nOri):
        
        # Select trials of a given orientation
        theseFrames = np.where(
            thisCharTrials['Orientation']==globalParams.ori[o]
            )[0]
        thisFluo_perOri = thisFluo[theseFrames]
        
        if boolIdenticalTuning:
            theseROI = np.where(
                thisCharROI['PrefOri']==globalParams.ori[o]
                )[0]
            thisFluo_perOri = thisFluo_perOri[:,theseROI]
            pairDist = np.triu(thisDistROI[theseROI,:][:,theseROI], 
                               k=1).flatten()
        
        # Mean-subtract the neural activity
        fluo_meanSub = thisFluo_perOri - np.mean(thisFluo_perOri, axis=0)
        
        # Full neural correlation matrix
        corr_full = np.corrcoef(fluo_meanSub, rowvar=False)
        
        # Set lower triangular and diagonal elements to zero
        pairCorr = np.triu(corr_full, k=1).flatten()
        
        # Select only the non-zero pairwise correlations
        tmpIdx = np.where(pairCorr!=0)[0]
        thisPairDist = pairDist[tmpIdx]
        thisPairCorr = pairCorr[tmpIdx]
        
        # Compute histogram
        for i in range(0, numBin):
            startEdge = edges[i]
            endEdge = edges[i+1]
            thisIdx = np.where((thisPairDist>startEdge) 
                               & (thisPairDist<=endEdge))
            if o==0:
                sortedPairCorr.append(thisPairCorr[thisIdx])
            else:
                sortedPairCorr[i] = np.concatenate((sortedPairCorr[i],
                                                    thisPairCorr[thisIdx]))
     
    # Mean and SEM of pairwise correlations for all orientations combined
    meanBinPairCorr = np.empty(numBin)
    meanBinPairCorr[:] = np.NaN
    semBinPairCorr = np.empty(numBin)
    semBinPairCorr[:] = np.NaN
    for i in range(0,numBin):
        meanBinPairCorr[i] = np.nanmean(sortedPairCorr[i])
        tmpStd = np.nanstd(sortedPairCorr[i])
        semBinPairCorr[i] = tmpStd / np.sqrt(len(sortedPairCorr[i]))

    # Plot errorbars
    plt.figure()
    plt.errorbar(binCenters, meanBinPairCorr, yerr=semBinPairCorr)
    plt.xlabel('Distance [um]')
    plt.ylabel('Correlation')
    #plt.xlim(0,250)
    plt.ylim(0,0.5)
    plt.grid(axis = 'y')
    plt.title('VisuallyEvoked=' + str(boolVisuallyEvoked) + ', OS=' + 
              str(boolOS) + ', IdenticalTuning=' + str(boolIdenticalTuning) +
              ', StimulusFrames=' + str(boolStimulus))
    #plt.savefig('corr_fdist.eps', format='eps')
    
    return binCenters, sortedPairCorr



# Plot pairwise correlation as a function of pairwise distance - as a function of the number of trials
def plot_ftrials_corr_fdist(fluo,distROI,nFramesPerTrial,binSize=10,nInstances=10,title=None):
    
    nTrials = int(fluo.shape[0]/nFramesPerTrial)
    
    # Vectors with number of randomly selected trials
    #numTrialsSel = np.arange(75,nTrials,50)
    numTrialsSel = np.round(np.linspace(75,nTrials,5)).astype(int)
    
    # Include the total number of trials
    if ~np.isin(nTrials,numTrialsSel):
        numTrialsSel = np.append(numTrialsSel,nTrials)
    
    numBin = int(500/binSize)
    edges = np.linspace(0,500,numBin+1)
    meanBinPairCorr = np.zeros((numBin,len(numTrialsSel),nInstances))
    semBinPairCorr = np.zeros((numBin,len(numTrialsSel),nInstances))
    meanBinPairCorr[:] = np.nan
    semBinPairCorr[:] = np.nan
    
    # Loop over the number of trials randomly selected
    for n in range(0,len(numTrialsSel)):
        
        # Loop over all the draw instances
        for draw in range(0,nInstances):
        
            # Number of trials selected
            thisNumTrialsSel = numTrialsSel[n]
            
            # Randomly select some trials
            theseTrials = rnd.sample(range(0,nTrials),thisNumTrialsSel)
            tmp = [i * nFramesPerTrial for i in theseTrials]
            tmp = [range(i,i+nFramesPerTrial) for i in tmp]
            tmp = np.reshape(np.array(tmp),thisNumTrialsSel*nFramesPerTrial)
            thisFluo = fluo[tmp,:]
            
            # Mean-subtract the neural activity
            fluo_meanSub = np.array(thisFluo - np.mean(thisFluo,axis=0))
            
            # Full neural covariance matrix
            corr_full = np.corrcoef(fluo_meanSub, rowvar=False)
            
            # Pairwise correlations
            pairCorr = np.triu(corr_full,k=1).flatten()
            
            # Pairwise distances
            pairDist = np.triu(distROI,k=1).flatten()
            
            # Select only the non-zero pairwise correlations
            tmpIdx = np.where(pairCorr!=0)
            pairDist = pairDist[tmpIdx]
            pairCorr = pairCorr[tmpIdx]
            
            # Compute histogram
            for i in range(0,numBin):
                startEdge = edges[i]
                endEdge = edges[i+1]
                #thisIdx = np.where((pairDist>startEdge) & (pairDist<=endEdge))
                thisIdx = np.squeeze(np.array(np.where((pairDist>startEdge) & (pairDist<=endEdge))))
                if thisIdx.size > 0:
                    meanBinPairCorr[i,n,draw] = np.mean(pairCorr[thisIdx])
                    semBinPairCorr[i,n,draw] = np.std(pairCorr[thisIdx])/np.sqrt(thisIdx.size)
    
    # Plot errorbars
    plt.figure()
    for n in range(0,len(numTrialsSel)):
        plt.errorbar(np.linspace(binSize/2,numBin*binSize-binSize/2,numBin), \
                     np.mean(meanBinPairCorr[:,n,:],axis=1), \
                         yerr=np.mean(semBinPairCorr[:,n,:],axis=1), \
                             label=str(numTrialsSel[n])+' trials')
    plt.xlabel('Distance [nm]')
    plt.ylabel('Correlation')
    plt.xlim(0,250)
    #plt.ylim(0,0.3)
    plt.grid(axis = 'y')
    plt.title(title)
    plt.legend()
    plt.show()
    

# Compute dimensionality for dataframe
def compute_dimensionalityDF(fluoDF):
    
    covDF = fluoDF.cov()
    u,s,__ = np.linalg.svd(covDF)
    this_PR = np.power(np.sum(s),2)/np.sum(np.power(s,2))
    this_N = fluoDF.shape[1]
    
    return this_PR,this_N



# Compute dimensionality of neuronal representation, assessed by Participation Ratio
def compute_dimensionality(fluo,type='full',boolPrint=False):
    
    # Mean-subtract the neural activity
    fluo_meanSub = np.array(fluo - np.nanmean(fluo,axis=0))
    
    
    if type=='full':
        #cov_type = np.matmul(np.transpose(fluo_meanSub), fluo_meanSub)
        cov_type = np.nancov(np.transpose(fluo_meanSub)) # full covariance matrix
        
    elif type=='shared':
        myFAmodel = FactorAnalysis(noise_variance_init=np.var(fluo_meanSub,axis=0)) #(noise_variance_init=np.var(fluo_meanSub,axis=0)) #(tol=1e-6,max_iter = 10000)
        
        # NB: This is the model:
        # C_full = W^T W + Psi = components_.T * components_ + diag(noise_variance)
        
        myFAmodel.fit(fluo_meanSub)
        instFR_FAcomp = myFAmodel.components_ # (n_components, n_features)
        
        cov_type = np.matmul(instFR_FAcomp.T, instFR_FAcomp) # shared covariance matrix
        
    elif type=='private':
        myFAmodel = FactorAnalysis(noise_variance_init=np.var(fluo_meanSub,axis=0)) #(noise_variance_init=np.var(fluo_meanSub,axis=0)) #(tol=1e-6,max_iter = 10000)
        
        # NB: This is the model:
        # C_full = W^T W + Psi = components_.T * components_ + diag(noise_variance)
        
        myFAmodel.fit(fluo_meanSub)
        
        cov_type = np.diag(myFAmodel.noise_variance_) # private covariance matrix
        
    # NB: these are two ways how to recover the full covariance matrix after Factor Analysis
    #cov_type = np.matmul(instFR_FAcomp.T, instFR_FAcomp) + np.diag(myFAmodel.noise_variance_) # full covariance matrix
    #cov_type = myFAmodel.get_covariance() # full covariance matrix
    
    # Perform Singluar value decomposition
    u,s,__ = np.linalg.svd(cov_type)
    
    # Compute Participation Ratio
    this_PR = np.power(np.sum(s),2)/np.sum(np.power(s,2))
    
    if boolPrint:
        if type=='full':
            print('Full participation ratio = '+ str(round(this_PR,1)) + ' (out of '+ str(fluo.shape[1]) +' ROIs)')
        elif type=='shared':
            print('Shared participation ratio = '+ str(round(this_PR,1)) + ' (out of '+ str(fluo.shape[1]) +' ROIs)')
        elif type=='private':
            print('Private participation ratio = '+ str(round(this_PR,1)) + ' (out of '+ str(fluo.shape[1]) +' ROIs)')
    
    
    return this_PR, cov_type, s



# Bootstrap of dimensionality computation
# nROIs: list of number of ROIs we want to sample, eg. nROIs=[5,10,20]
# nTimes: number of times we want to sample
def compute_dimensionalityBootstrap(fluo,nROIs=[10],nTimes=10,type='full',boolPlot=True):

    # Mean-subtract the neural activity
    fluo_meanSub = np.array(fluo - np.mean(fluo,axis=0))
    
    # Create empy list to fill with the PR values
    allPR = [[0] * nTimes for i in range(len(nROIs))]
    
    for nr in range(len(nROIs)):
        thisNR = nROIs[nr]
        
        for t in range(nTimes):
            theseROIs = rnd.sample(range(0,fluo.shape[1]),thisNR)
            
            thisFluoMeanSub = fluo_meanSub[:,theseROIs]
            
            if type=='full':
                cov_type = np.matmul(np.transpose(thisFluoMeanSub), thisFluoMeanSub)
            elif type=='shared':
                myFAmodel = FactorAnalysis(noise_variance_init=np.var(thisFluoMeanSub,axis=0))
                
                # NB: This is the model:
                # C_full = W^T W + Psi = components_.T * components_ + diag(noise_variance)
                
                myFAmodel.fit(thisFluoMeanSub)
                instFR_FAcomp = myFAmodel.components_ # (n_components, n_features)
                
                cov_type = np.matmul(instFR_FAcomp.T, instFR_FAcomp) # shared covariance matrix
                
            # Perform Singluar value decomposition
            u,s,__ = np.linalg.svd(cov_type)
            
            # Compute Participation Ratio...
            thisPR = np.power(np.sum(s),2)/np.sum(np.power(s,2))
            # ... and write it down
            allPR[nr][t] = thisPR
    
    if boolPlot==True:
        meanPR = np.mean(np.array(allPR),axis=1)
        semPR = np.std(np.array(allPR),axis=1)/np.sqrt(nTimes)
        plt.figure()
        plt.errorbar(nROIs,meanPR,yerr=semPR)
        plt.xlabel('Number of ROIs')
        plt.xticks(nROIs)
        #plt.ylim((0,25))
        if type=='full':
            plt.ylabel('Full PR')
        elif type=='shared':
            plt.ylabel('Shared PR')
        plt.title('Sampling ROIs '+str(nTimes)+' times')
        
                
    return allPR


### new
# Bootstrap of dimensionality computation
# nROIs: list of number of ROIs we want to sample, eg. nROIs=[5,10,20]
# nTimes: number of times we want to sample
def compute_dimensionalityBootstrap_spontVSevoked(fluo,idxBlank,idxStimulus,nROIs=[10],nTimes=10,type='full',boolPlot=True):
    
    plt.figure()
    for thoseF in range(2):
        
        if thoseF==0:
            thisFluo = fluo[idxBlank,:]
        elif thoseF==1:
            thisFluo = fluo[idxStimulus,:]
            
        # Mean-subtract the neural activity
        fluo_meanSub = np.array(thisFluo - np.mean(thisFluo,axis=0))
        
        # Create empy list to fill with the PR values
        allPR = [[0] * nTimes for i in range(len(nROIs))]
        
        for nr in range(len(nROIs)):
            thisNR = nROIs[nr]
            
            for t in range(nTimes):
                theseROIs = rnd.sample(range(0,thisFluo.shape[1]),thisNR)
                
                thisFluoMeanSub = fluo_meanSub[:,theseROIs]
                
                if type=='full':
                    cov_type = np.matmul(np.transpose(thisFluoMeanSub), thisFluoMeanSub)
                    #cov_type = np.corrcoef(np.transpose(thisFluoMeanSub))
                elif type=='shared':
                    myFAmodel = FactorAnalysis(noise_variance_init=np.var(thisFluoMeanSub,axis=0))
                    
                    # NB: This is the model:
                    # C_full = W^T W + Psi = components_.T * components_ + diag(noise_variance)
                    
                    myFAmodel.fit(thisFluoMeanSub)
                    instFR_FAcomp = myFAmodel.components_ # (n_components, n_features)
                    
                    cov_type = np.matmul(instFR_FAcomp.T, instFR_FAcomp) # shared covariance matrix
                    
                # Perform Singluar value decomposition
                u,s,__ = np.linalg.svd(cov_type)
                
                # Compute Participation Ratio...
                thisPR = np.power(np.sum(s),2)/np.sum(np.power(s,2))
                # ... and write it down
                allPR[nr][t] = thisPR
                
        if thoseF==0:
            allPR_Blank = allPR
        elif thoseF==1:
            allPR_Stimulus = allPR
    
        if boolPlot==True:
            meanPR = np.mean(np.array(allPR),axis=1)
            semPR = np.std(np.array(allPR),axis=1)/np.sqrt(nTimes)
            
            plt.errorbar(nROIs,meanPR/nROIs,yerr=semPR)
            plt.xlabel('Number of ROIs')
            plt.xticks(nROIs)
            #plt.ylim((0,25))
            plt.ylabel(type+' PR/N')
            # if type=='full':
            #     plt.ylabel('Full PR')
            # elif type=='shared':
            #     plt.ylabel('Shared PR')
            plt.title('Sampling ROIs '+str(nTimes)+' times')
        
                
    return allPR_Blank,allPR_Stimulus


# Plot the average fluorescence over all trial orientations and all ROIs
def plot_avgFluo(dataFramesPerTrial,charTrials,fluo,frameRate):
    
    tmpTime = (1/frameRate)*(np.arange(dataFramesPerTrial)-globalParams.nBlankFrames)
        
    charTrials = charTrials[['Orientation','TrialFrame']]
    data = pd.concat([charTrials, fluo],axis=1)
    
    avgPerOri = data.groupby(['TrialFrame','Orientation']).mean()
    avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
    avgPerOri = avgPerOri.reset_index()

    tmp = avgPerOri
    tmp = np.array(tmp.drop(['TrialFrame','Orientation'],axis=1))
    tmp = np.reshape(tmp,(dataFramesPerTrial,-1),order='F')
    tmpMean = np.mean(tmp,axis=1)
    tmpSEM = np.std(tmp,axis=1)/np.sqrt(tmp.shape[1])
    tmpStd = np.std(tmp,axis=1)

    plt.figure()
    plt.fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    plt.plot(tmpTime,tmpMean,color='k')
    plt.title('Average fluorescence')   

    plt.figure()
    plt.plot(tmpTime,tmpStd/tmpMean,color='k')
    plt.title('Average coefficient of variation')             
                

# Plot the average fluorescence for each trial orientation
def plot_avgFluoPerOri(dataFramesPerTrial,charTrials,fluo,frameRate,title=None):
    
    tmpTime = (1/frameRate)*(np.arange(dataFramesPerTrial)-globalParams.nBlankFrames)
        
    charTrials = charTrials[['Orientation','TrialFrame']]
    data = pd.concat([charTrials, fluo],axis=1)
    
    avgPerOri = data.groupby(['TrialFrame','Orientation']).mean()
    avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
    avgPerOri = avgPerOri.reset_index()
    
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for o in range(4):
        tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[o]]
        tmp = np.array(tmp.drop(['TrialFrame','Orientation'],axis=1))
        tmpMean = np.mean(tmp,axis=1)
        tmpSEM = np.std(tmp,axis=1)/np.sqrt(tmp.shape[1])
        axs[int(o/2), np.mod(o,2)].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
        axs[int(o/2), np.mod(o,2)].plot(tmpTime,tmpMean,color='k')
        axs[int(o/2), np.mod(o,2)].set_title('Orientation: '+str(globalParams.ori[o])+'°')
    fig.suptitle(title)
    
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for o in range(4):
        tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[o]]
        tmp = np.array(tmp.drop(['TrialFrame','Orientation'],axis=1))
        tmpMean = np.mean(tmp,axis=1)
        tmpStd = np.std(tmp,axis=1)
        axs[int(o/2), np.mod(o,2)].plot(tmpTime,tmpStd/tmpMean,color='k')
        axs[int(o/2), np.mod(o,2)].set_title('Orientation: '+str(globalParams.ori[o])+'°')
    fig.suptitle('Coefficient of variation')


# Plot the average fluorescence for each trial orientation for each ROI
def plot_avgFluoPerOriPerROI(dataFramesPerTrial,charTrials,fluo,frameRate,title=None):
    
    tmpTime = (1/frameRate)*(np.arange(dataFramesPerTrial)-globalParams.nBlankFrames)
        
    charTrials = charTrials[['Orientation','TrialFrame']]
    data = pd.concat([charTrials, fluo],axis=1)
    
    avgPerOri = data.groupby(['TrialFrame','Orientation']).mean()
    avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
    avgPerOri = avgPerOri.reset_index()
    
    semPerOri = data.groupby(['TrialFrame','Orientation']).sem()
    semPerOri = semPerOri.sort_values(['Orientation','TrialFrame'])
    semPerOri = semPerOri.reset_index()
    
    stdPerOri = data.groupby(['TrialFrame','Orientation']).std()
    stdPerOri = stdPerOri.sort_values(['Orientation','TrialFrame'])
    stdPerOri = stdPerOri.reset_index()
    
    sqrtNP = 5
    fig, axs = plt.subplots(sqrtNP, sqrtNP, sharex=True, sharey=True)
    #fig, axs = plt.subplots(sqrtNP, sqrtNP)
    for n in range(sqrtNP*sqrtNP):
        for o in range(4):
            tmp = avgPerOri[n].loc[avgPerOri['Orientation']==globalParams.ori[o]]
            tmpSEM = semPerOri[n].loc[semPerOri['Orientation']==globalParams.ori[o]]
            axs[int(n/sqrtNP), np.mod(n,sqrtNP)].fill_between(tmpTime,tmp-tmpSEM,tmp+tmpSEM)
            axs[int(n/sqrtNP), np.mod(n,sqrtNP)].plot(tmpTime,tmp)
    fig.suptitle('Fluorescence: mean +/- SEM')

    fig, axs = plt.subplots(sqrtNP, sqrtNP, sharex=True, sharey=True)
    #fig, axs = plt.subplots(sqrtNP, sqrtNP)
    for n in range(sqrtNP*sqrtNP):
        for o in range(4):
            tmpSEM = semPerOri[n].loc[semPerOri['Orientation']==globalParams.ori[o]]
            axs[int(n/sqrtNP), np.mod(n,sqrtNP)].plot(tmpTime,tmpSEM)
    fig.suptitle('SEM')
    
    fig, axs = plt.subplots(sqrtNP, sqrtNP, sharex=True, sharey=True)
    #fig, axs = plt.subplots(sqrtNP, sqrtNP)
    for n in range(sqrtNP*sqrtNP):
        for o in range(4):
            tmp = avgPerOri[n].loc[avgPerOri['Orientation']==globalParams.ori[o]]
            tmpStd = stdPerOri[n].loc[stdPerOri['Orientation']==globalParams.ori[o]]
            axs[int(n/sqrtNP), np.mod(n,sqrtNP)].plot(tmpTime,tmpStd/tmp)
    fig.suptitle('Coefficient of variation')
      

# Plot the average behavioral trace over all trials
def plot_avgBehavioralTrace(dataType,charTrials,frameRate,boolMotion,boolPupil):
    
    if dataType=='L4_cytosolic':
        tmpTime = (1/frameRate)*(np.arange(25)-5) # np.arange(25)+1
    elif dataType=='L23_thalamicBoutons':
        tmpTime = (1/frameRate)*(np.arange(30)-5) # np.arange(30)+1
      
    if boolMotion:
        data = charTrials[['TrialFrame','motSVD1']]
        
        tmpMean = np.squeeze(np.array(data.groupby(['TrialFrame']).mean()))
        tmpSEM = np.squeeze(np.array(data.groupby(['TrialFrame']).std()/np.sqrt(data.shape[0]/len(tmpTime))))
        
        plt.figure()
        plt.fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
        plt.plot(tmpTime,tmpMean,color='k')
        plt.xlabel('Aligned time [s]')
        plt.title('Average motSVD1 over all trials')
        
    if boolPupil:
        data = charTrials[['TrialFrame','pupilArea']]
        
        tmpMean = np.squeeze(np.array(data.groupby(['TrialFrame']).mean()))
        tmpSEM = np.squeeze(np.array(data.groupby(['TrialFrame']).std()/np.sqrt(data.shape[0]/len(tmpTime))))
        
        plt.figure()
        plt.fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
        plt.plot(tmpTime,tmpMean,color='k')
        plt.xlabel('Aligned time [s]')
        plt.title('Average pupilArea over all trials')  


# Compute Fano factor for each ROIs
def compute_fanoFactor(dataType,charTrials,fluo,title=None):
    
    fluo = np.array(fluo)
    nROIs = fluo.shape[1]
    nFrames = fluo.shape[0]
    
    if dataType=='L4_cytosolic':
        tmpTime = np.arange(25)+1
    elif dataType=='L23_thalamicBoutons':
        tmpTime = np.arange(30)+1
    
    # Compute Fano Factor over all trials for each ROI
    fanoFactor = []
    for n in range(0,nROIs):
        tmpFluo = fluo[:,n]
        tmpFluo = np.reshape(tmpFluo,(int(nFrames/len(tmpTime)),len(tmpTime)))
        meanFluo = np.mean(tmpFluo,axis=0)
        varFluo = np.var(tmpFluo,axis=0)
        fanoFactor.append(varFluo/meanFluo)
    
    meanFF = np.nanmean(np.array(fanoFactor),axis=0)
    semFF = np.nanstd(np.array(fanoFactor),axis=0)/np.sqrt(nROIs)
    
    plt.figure()
    plt.plot(tmpTime,meanFF,'k-')
    plt.fill_between(tmpTime, meanFF-semFF, meanFF+semFF)
    plt.xlabel('Aligned time frames')
    plt.ylabel('Average Fano Factor')
    #plt.ylim((0.8,1.9))
    plt.title(title)
    
    # Compute Fano Factor over all trials of each orientation separately for each ROI
    for o in range(4):
        fanoFactorPerThisOri = []
        tmpIdx = np.where(charTrials['Orientation']==globalParams.ori[o])[0]
        tmpFluo = fluo[tmpIdx,:]
        nFrames = len(tmpIdx)
        for n in range(0,nROIs):
            tmpFluo1ROI = tmpFluo[:,n]
            tmpFluo1ROI = np.reshape(tmpFluo1ROI,(int(nFrames/len(tmpTime)),len(tmpTime)))
            meanFluo = np.mean(tmpFluo1ROI,axis=0)
            varFluo = np.var(tmpFluo1ROI,axis=0)
            fanoFactorPerThisOri.append(varFluo/meanFluo)
        fanoFactorPerThisOri = np.array(fanoFactorPerThisOri)
        if o==0:
            fanoFactorPerOri = fanoFactorPerThisOri
            fanoFactorPerOri = fanoFactorPerOri[:,:,np.newaxis]
        else:
            fanoFactorPerOri = np.concatenate((fanoFactorPerOri,fanoFactorPerThisOri[:,:,np.newaxis]),axis=2)
    
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    for o in range(4):
        tmp = fanoFactorPerOri[:,:,o]
        tmpMean = np.mean(tmp,axis=0)
        tmpSEM = np.std(tmp,axis=0)/np.sqrt(tmp.shape[0])
        axs[int(o/2), np.mod(o,2)].plot(tmpTime,tmpMean)
        axs[int(o/2), np.mod(o,2)].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
        axs[int(o/2), np.mod(o,2)].set_title('Orientation: '+str(globalParams.ori[o])+'°')
    fig.suptitle(title)
    
    
    return np.array(fanoFactor), fanoFactorPerOri






