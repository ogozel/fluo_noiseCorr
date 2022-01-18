# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:58:45 2021

@author: Olivia Gozel

Functions to do the analyses
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import statsmodels.tsa.stattools as smt
import random as rnd



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
    #s = np.square(pca.singular_values_)

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
    plt.ylim(4,20)
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
    plt.ylim(0,30)
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
    plt.ylim(0,0.20)
    plt.grid(axis = 'y')
    plt.title(title)
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
    #plt.ylim(0,0.3)
    plt.grid(axis = 'y')
    plt.title(title)
    plt.legend()
    plt.show()
    
    
    
    
    
    