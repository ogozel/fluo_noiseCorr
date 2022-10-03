# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:23:57 2021

@author: Olivia Gozel

Functions to do the preprocessing
"""

import h5py
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.io
from scipy.ndimage import center_of_mass
import scipy.stats as stats
from sklearn.metrics.pairwise import euclidean_distances

import os
os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams




def loadData(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub,nBlankFrames,nStimFrames):
    
    nTrialsPerSession = np.zeros(len(dataSessions),dtype=int)
    for s in range(len(dataSessions)):
    
        sessionNumber = dataSessions[s]
        dataFramesPerTrial = nBlankFrames + nStimFrames
        
        filepath = []
        if dataType == 'L4_cytosolic':
            filepath.append(globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
                dataMouse + '\\' + dataDepth + '\\S' + str(sessionNumber) + \
                    '\\ROI_' + dataDate + '_RM_S' + str(sessionNumber) + \
                        '_Intensity_unweighted_s2p_' + dataNeuropilSub + '.mat')
            # filepath.append(globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
            #     dataMouse + '\\' + dataDepth + '\\Updated_July2022\\' + \
            #         '\\ROI_' + dataDate + '_S' + str(sessionNumber) + \
            #             '_Intensity_unweighted_imj_' + dataNeuropilSub + '.mat')
                
        elif dataType == 'L23_thalamicBoutons':
            filepath.append(globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
                                dataMouse + '\\' + dataDepth + '\\L23\\ROI_' + \
                                    '' + dataDate + '_' + dataDepth + '_S' + str(sessionNumber) + \
                                            '_Intensity_unweighted_' + dataNeuropilSub + '.mat')
        
        
        this_dff0 = []
        this_order = []
        this_positionROI3d = []

        arrays = {}
        f = h5py.File(filepath[0],'r')
        for k, v in f.items():
            arrays[k] = np.array(v)
        
        
        this_order = arrays['order'][0] # the orientation for each frame (also blank frames)
        if dataType == 'L4_cytosolic':
            this_dff0 = arrays['dff0'] # fluorescence data
        elif dataType == 'L23_thalamicBoutons':
            this_dff0 = arrays['dff0_allstacks'] # fluorescence data
        nTrialsPerSession[s] = int(this_dff0.shape[1]/dataFramesPerTrial)
        this_positionROI3d = arrays['bw']
        
        # Concatenate the data
        if s==0:
            dff0 = this_dff0
            order = this_order
        else:
            dff0 = np.concatenate((dff0,this_dff0),axis=1)
            order = np.concatenate((order,this_order),axis=0)
            
        # Save info about the ROIs for the first session only
        if s==0:
            positionROI_3d = this_positionROI3d
    
    return dff0,order,positionROI_3d,nTrialsPerSession



def determine_params(dff0,positionROI_3d,nBlankFrames,nStimFrames):
    
    nROI_init = dff0.shape[0] # number of neurons before preprocessing
    nFrames = dff0.shape[1] # number of fluorescence frames
    dataFramesPerTrial = nBlankFrames + nStimFrames
    nTrials = int(nFrames/dataFramesPerTrial)
    fluoPlaneWidth = positionROI_3d.shape[1]
    fluoPlaneHeight = positionROI_3d.shape[2]
    
    return nROI_init,nTrials,fluoPlaneWidth,fluoPlaneHeight



def selectROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,order,dff0,positionROI_3d,nBlankFrames,nStimFrames):
    
    # Parameters
    nROI_init = dff0.shape[0] # number of neurons before preprocessing
    nFrames = dff0.shape[1] # number of fluorescence frames
    dataFramesPerTrial = nBlankFrames + nStimFrames
    nTrials = int(nFrames/dataFramesPerTrial)
    fluoPlaneWidth = positionROI_3d.shape[1]
    fluoPlaneHeight = positionROI_3d.shape[2]
    
    ### start
    # (1) Set 'inf' dff0 values to 'nan'
    tmpBoolInf = np.isinf(dff0)
    dff0[tmpBoolInf] = np.nan
    
    # (2) Set dff0 values > 1000% or < 1000% to 'nan'
    tmpBoolTooHigh = (dff0>1000)
    tmpBoolTooLow = (dff0<-1000)
    dff0[tmpBoolTooHigh] = np.nan
    dff0[tmpBoolTooLow] = np.nan
    
    # (3) Discard boutons that have more than 25% of dff0 values which are 'nan'
    tmpBoolNan = np.isnan(dff0)
    tmpSum = np.sum(tmpBoolNan,axis=1)
    tmpIdx = np.where((tmpSum/nFrames) < 0.25)[0]
    dff0 = dff0[tmpIdx,:]
    positionROI_3d = positionROI_3d[tmpIdx]
    
    # Number of ROIs we keep
    nROI = dff0.shape[0]
    print('We intially keep '+str(nROI)+' boutons out of '+str(nROI_init)+' recorded boutons ('+str(np.round(100*nROI/nROI_init))+'%).')
    nROI_init = nROI
    ### end
    
    # Create a DataFrame for the ROI positions
    positionROI = pd.DataFrame(positionROI_3d.reshape(nROI_init,fluoPlaneWidth*fluoPlaneHeight))
    
    # Create a DataFrame for the fluorescence data
    fluo_array = np.transpose(dff0)
    fluo = pd.DataFrame(np.transpose(dff0))

    # Create a DataFrame for the ROI characteristics
    charROI = pd.DataFrame()
    
    # Create a DataFrame for the trial characteristics
    charTrials = pd.DataFrame(np.repeat(dataSessions,dataFramesPerTrial*(nTrialsPerSession)),columns=['Session'])
    charTrials['Trial'] = np.repeat(np.arange(nTrials),dataFramesPerTrial)
    charTrials['TrialFrame'] = np.tile(np.arange(dataFramesPerTrial),nTrials)
    
    tmp = np.concatenate((np.repeat(['Blank'],nBlankFrames),\
                          np.repeat(['Stimulus'],nStimFrames))) 
    frameType = np.tile(tmp,nTrials)
    charTrials['FrameType'] = frameType
    charTrials['Orientation'] = order
    
    data = pd.concat([charTrials,fluo],axis=1)
    
    # Average fluorescence value for blank frames and stimulus frames for each trial
    avgFluo_Blank = data.loc[data['FrameType']=='Blank'].groupby(['Orientation','Trial']).mean()
    avgFluo_Blank = avgFluo_Blank.droplevel(level=1)
    avgFluo_Blank = avgFluo_Blank.drop(columns=['Session','TrialFrame'])
    avgFluo_Stimulus = data.loc[data['FrameType']=='Stimulus'].groupby(['Orientation','Trial']).mean()
    avgFluo_Stimulus = avgFluo_Stimulus.droplevel(level=1)
    avgFluo_Stimulus = avgFluo_Stimulus.drop(columns=['Session','TrialFrame'])
    
    # Anova test to determine which ROIs are visually responsive (=are activated by the grating stimuli)
    bool_visuallyResp = np.zeros((nROI_init,globalParams.nOri))
    for o in range(globalParams.nOri):
        thisB = avgFluo_Blank.loc[globalParams.ori[o]]
        thisG = avgFluo_Stimulus.loc[globalParams.ori[o]]
        F,pval = stats.f_oneway(thisB,thisG)
        bool_visuallyResp[:,o] = (pval < globalParams.threshold_pval_visuallyEvoked)
    
    idxVE = np.where(np.sum(bool_visuallyResp,axis=1) > 0)[0]
    charROI['VisuallyEvoked'] =  np.array([True if i in idxVE else False for i in range(nROI_init)])
    
    # Orientation preference
    tmpAvg = avgFluo_Stimulus.groupby('Orientation').mean()
    charROI['PrefOri'] = tmpAvg.idxmax(axis=0)
    
    print(str(len(np.where(charROI['VisuallyEvoked']==True)[0]))+' out of '+str(nROI_init)+' ROIs are visually responsive ('+str(np.round(100*len(np.where(charROI['VisuallyEvoked']==True)[0])/nROI_init).astype(int))+'%)')
    
    # Center of mass of ROIs (scipy.ndimage.center_of_mass)
    for n in range(nROI_init):
        this_xCM, this_yCM = center_of_mass(positionROI_3d[n])
        this_xCM = int(np.round(this_xCM))
        this_yCM = int(np.round(this_yCM))
        if n==0:
            cmROI = np.array([[this_xCM,this_yCM]])
        else:
            cmROI = np.append(cmROI,np.array([[this_xCM,this_yCM]]),axis=0)
            
    charROI['xCM'] = pd.DataFrame(cmROI[:,0])
    charROI['yCM'] = pd.DataFrame(cmROI[:,1])
    
    # Compute pairwise distance between all ROIs
    distROI = pixelSize*euclidean_distances(cmROI)
    
    # Find pairs of ROIs which are too close to each other
    tmp_idxROItooClose = np.argwhere(distROI<globalParams.threshold_distance)
    
    # Size of ROIs (unit: pixels)
    charROI['Size'] = positionROI.sum(axis=1)
    
    # Find the ROI which are the smallest of a pair which is too close (to discard)
    idxROItooClose = []
    for i in range(tmp_idxROItooClose.shape[0]):
        this_pair = tmp_idxROItooClose[i,:]
        if this_pair[0] != this_pair[1]:
            if ~np.isin(this_pair[0],idxROItooClose) & ~np.isin(this_pair[1],idxROItooClose):
                tmpSize0 = charROI['Size'][this_pair[0]]
                tmpSize1 = charROI['Size'][this_pair[1]]
                if tmpSize0 < tmpSize1:
                    idxROItooClose = np.append(idxROItooClose,this_pair[0])
                else:
                    idxROItooClose = np.append(idxROItooClose,this_pair[1])
    
    # Are ROIs far enough? (if not: discard smallest one of the pair)
    print('There are '+str(len(idxROItooClose))+' ROIs which are discarded because they are too close from another (bigger) ROI')
    charROI['farEnough'] = np.array([False if i in idxROItooClose else True for i in range(nROI_init)])
    
    # Anova test to determine orientation selectivity
    group0 = avgFluo_Stimulus.loc[globalParams.ori[0]]
    group1 = avgFluo_Stimulus.loc[globalParams.ori[1]]
    group2 = avgFluo_Stimulus.loc[globalParams.ori[2]]
    group3 = avgFluo_Stimulus.loc[globalParams.ori[3]]
    f_val, p_val = stats.f_oneway(group0,group1,group2,group3)
    bool_OS = (p_val < globalParams.threshold_pval_OS)
    
    # Not visually evoked ROIs cannot be orientation selective
    idxNotVE = np.where((charROI['VisuallyEvoked']==False)&(bool_OS==True))[0]
    bool_OS[idxNotVE] = 0
    
    # Are ROIs orientation-selective
    nOS = len(np.where(bool_OS==True)[0])
    nVE = len(np.where(charROI['VisuallyEvoked']==True)[0])
    print('There are '+str(nOS)+' visually evoked ROIs which are orientation selective ('+str(np.round(100*nOS/nVE).astype(int))+'%)')
    charROI['OS'] = np.array([True if bool_OS[i]==1 else False for i in range(nROI_init)])
    
    # Determine which ROIs we keep for the analyses
    charROI['keptROI'] = charROI['farEnough']
    
    # Plot all ROIs
    plt.figure()
    plt.imshow(np.transpose(np.sum(positionROI_3d,axis=0)))
    plt.title('All ROIs')
    
    idxNotVR = np.where(charROI['VisuallyEvoked']==False)[0]
    plt.figure()
    plt.imshow(np.transpose(np.sum(positionROI_3d[idxNotVR],axis=0)))
    plt.title('Not visually-evoked ROIs')
    
    # Discard bad ROIs
    idxKept = np.where(charROI['keptROI']==True)[0]
    fluo_array = fluo_array[:,idxKept]
    fluo = pd.DataFrame(fluo_array)
    positionROI = pd.DataFrame(np.array(positionROI)[idxKept,:])
    positionROI_3d = positionROI_3d[idxKept,:,:]
    distROI = distROI[idxKept,:][:,idxKept]
    charROI = charROI[charROI['keptROI']==True].reset_index()
    charROI = charROI.drop(columns=['index','farEnough','keptROI'])
    data = pd.concat([charTrials, fluo],axis=1)
    
    # Number of ROIs we keep after preprocessing
    nROI = charROI.shape[0]

    # Plot all kept ROIs at the end of preprocessing
    plt.figure()
    plt.imshow(np.transpose(np.sum(positionROI_3d,axis=0)))
    plt.title('ROIs kept after postprocessing')
    
    # Plot all kept ROIs with their selectivity
    plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)
    
    # Plot the average fluorescence over all ROIs for each trial orientation
    plot_avgFluoPerOri(data,dataFramesPerTrial)
    
    # Plot the average fluorescence over only the OS ROIs for each trial orientation
    idxOS = np.where(charROI['OS']==True)[0]
    fluo_array = fluo_array[:,idxOS]
    fluoOS = pd.DataFrame(fluo_array)
    dataOS = pd.concat([charTrials, fluoOS],axis=1)
    plot_avgFluoPerOri(dataOS,dataFramesPerTrial)
    
    print('We keep '+str(nROI)+' ROIs out of '+str(nROI_init)+' recorded ROIs ('+str(np.round(100*nROI/nROI_init).astype(int))+'%).')
        
    return charROI,charTrials,fluo,data,positionROI,distROI,idxKept


# # Plot all kept ROIs at the end of preprocessing with their selectivity
# def plot_ROIwithSelectivity(charROI,positionROI_3d):
    
#     fluoPlaneWidth = positionROI_3d.shape[1]
#     fluoPlaneHeight = positionROI_3d.shape[2]
    
#     coloredROI = np.zeros((fluoPlaneHeight,fluoPlaneWidth))
#     for i in range(len(globalParams.ori)):
#         tmpIdx = np.squeeze(np.array(np.where((charROI['OS']==True) & (charROI['PrefOri']==globalParams.ori[i]))))
#         if tmpIdx.size==1:
#             tmp = np.clip((i+1)*positionROI_3d[tmpIdx],0,i+1)
#         else:
#             tmp = np.clip(np.sum((i+1)*positionROI_3d[tmpIdx],axis=0),0,i+1)
#         coloredROI = np.clip(coloredROI + tmp,0,i+1)
#         #plt.figure()
#         #plt.imshow(np.transpose(np.sum((i+1)*positionROI_3d[tmpIdx],axis=0)))
    
#     # Non-OS kept ROIs
#     tmpIdx = np.squeeze(np.array(np.where((charROI['OS']==False))))
#     if tmpIdx.size==1:
#         tmp = np.clip((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],0,len(globalParams.ori)+1)
#     else:
#         tmp = np.clip(np.sum((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],axis=0),0,len(globalParams.ori)+1)
#     coloredROI = np.clip(coloredROI + tmp, 0, len(globalParams.ori)+1)
#     #plt.figure()
#     #plt.imshow(np.transpose(np.sum((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],axis=0)))
    
#     # Clip the values for overlaps
#     coloredROI = np.clip(coloredROI,0,len(globalParams.ori)+1)
    
#     # Create colormap
#     cmap = colors.ListedColormap(['white', 'red', 'green', 'blue', 'orange', 'black'])
#     bounds = np.linspace(0,6,7)
#     norm = colors.BoundaryNorm(bounds, cmap.N)
    
#     plt.figure()
#     # tell imshow about color map so that only set colors are used
#     img = plt.imshow(np.transpose(coloredROI), interpolation='nearest', cmap=cmap, norm=norm)
    
#     # make a color bar (it uses 'cmap' and 'norm' form above)
#     cbar = plt.colorbar(img, boundaries=np.linspace(1,6,6), ticks=np.linspace(1.5,5.5,5))
#     cbar.ax.set_yticklabels(['45°', '135°','180°','270°','non-OS'])
    
#     plt.title('Kept ROIs with orientation selectivity')


# Plot all kept ROIs at the end of preprocessing with their selectivity
# def plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight):
    
#     positionROI = np.array(positionROI)
#     positionROI_3d = positionROI.reshape((positionROI.shape[0],fluoPlaneWidth,fluoPlaneHeight))
    
#     coloredROI = np.zeros((fluoPlaneHeight,fluoPlaneWidth))
#     for i in range(len(globalParams.ori)):
#         tmpIdx = np.squeeze(np.array(np.where((charROI['OS']==True) & (charROI['PrefOri']==globalParams.ori[i]))))
#         if tmpIdx.size==1:
#             tmp = np.clip((i+1)*positionROI_3d[tmpIdx],0,i+1)
#         else:
#             tmp = np.clip(np.sum((i+1)*positionROI_3d[tmpIdx],axis=0),0,i+1)
#         coloredROI = np.clip(coloredROI + tmp,0,i+1)
#         #plt.figure()
#         #plt.imshow(np.transpose(np.sum((i+1)*positionROI_3d[tmpIdx],axis=0)))
    
#     # Non-OS kept ROIs
#     tmpIdx = np.squeeze(np.array(np.where((charROI['OS']==False))))
#     if tmpIdx.size==1:
#         tmp = np.clip((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],0,len(globalParams.ori)+1)
#     else:
#         tmp = np.clip(np.sum((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],axis=0),0,len(globalParams.ori)+1)
#     coloredROI = np.clip(coloredROI + tmp, 0, len(globalParams.ori)+1)
#     #plt.figure()
#     #plt.imshow(np.transpose(np.sum((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],axis=0)))
    
#     # Clip the values for overlaps
#     coloredROI = np.clip(coloredROI,0,len(globalParams.ori)+1)
    
#     # Create colormap
#     cmap = colors.ListedColormap(['white', 'red', 'green', 'blue', 'orange', 'black'])
#     bounds = np.linspace(0,6,7)
#     norm = colors.BoundaryNorm(bounds, cmap.N)
    
#     plt.figure()
#     # tell imshow about color map so that only set colors are used
#     img = plt.imshow(np.transpose(coloredROI), interpolation='nearest', cmap=cmap, norm=norm)
    
#     # make a color bar (it uses 'cmap' and 'norm' form above)
#     cbar = plt.colorbar(img, boundaries=np.linspace(1,6,6), ticks=np.linspace(1.5,5.5,5))
#     cbar.ax.set_yticklabels(['45°', '135°','180°','270°','non-OS'])
    
#     plt.title('Kept ROIs with orientation selectivity')
#     #plt.savefig('ROIwithSel.eps', format='eps')


def plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight):
    
    positionROI = np.array(positionROI)
    positionROI_3d = positionROI.reshape((positionROI.shape[0],fluoPlaneWidth,fluoPlaneHeight))
    
    coloredROI = np.zeros((fluoPlaneHeight,fluoPlaneWidth))
    for i in range(len(globalParams.ori)):
        tmpIdx = np.squeeze(np.array(np.where((charROI['OS']==True) & (charROI['PrefOri']==globalParams.ori[i]))))
        if tmpIdx.size==1:
            tmp = np.clip((i+1)*positionROI_3d[tmpIdx],0,i+1)
        else:
            tmp = np.clip(np.sum((i+1)*positionROI_3d[tmpIdx],axis=0),0,i+1)
        coloredROI = np.clip(coloredROI + tmp,0,i+1)
    
    # Visually responsive non-OS ROIs
    tmpIdx = np.where((charROI['VisuallyEvoked']==True)&(charROI['OS']==False))[0]
    if tmpIdx.size==1:
        tmp = np.clip((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],0,len(globalParams.ori)+1)
    else:
        tmp = np.clip(np.sum((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],axis=0),0,len(globalParams.ori)+1)
    coloredROI = np.clip(coloredROI + tmp, 0, len(globalParams.ori)+1)
    
    # Clip the values for overlaps
    coloredROI = np.clip(coloredROI,0,len(globalParams.ori)+1)
    
    # Non-visually responsive ROIs (=spontaneously active ROIs)
    tmpIdx = np.where(charROI['VisuallyEvoked']==False)[0]
    if tmpIdx.size==1:
        tmp = np.clip((len(globalParams.ori)+2)*positionROI_3d[tmpIdx,:,:],0,len(globalParams.ori)+2)
    else:
        tmp = np.clip(np.sum((len(globalParams.ori)+2)*positionROI_3d[tmpIdx,:,:],axis=0),0,len(globalParams.ori)+2)
    coloredROI = np.clip(coloredROI + tmp, 0, len(globalParams.ori)+2)
    
    # Create colormap
    cmap = colors.ListedColormap(['white', 'red', 'green', 'blue', 'orange', 'black', 'grey'])
    bounds = np.linspace(0,7,8)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure()
    # tell imshow about color map so that only set colors are used
    img = plt.imshow(np.transpose(coloredROI), interpolation='nearest', cmap=cmap, norm=norm)
    
    # make a color bar (it uses 'cmap' and 'norm' form above)
    cbar = plt.colorbar(img, boundaries=np.linspace(1,7,7), ticks=np.linspace(1.5,6.5,6))
    cbar.ax.set_yticklabels(['45°', '135°','180°','270°','non-OS','spont'])

    
    plt.title('Kept ROIs with orientation selectivity')
    #plt.savefig('ROIwithSel.eps', format='eps')



# Plot all kept V1 ROIs and thalamic axons at the end of preprocessing
def plot_ROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight,idxOverlappingBoutons=None):
    
    positionROI = np.array(positionROI)
    positionBoutons = np.array(positionBoutons)
    positionROI_3d = positionROI.reshape((positionROI.shape[0],fluoPlaneWidth,fluoPlaneHeight))
    positionBoutons_3d = positionBoutons.reshape((positionBoutons.shape[0],fluoPlaneWidth,fluoPlaneHeight))
    
    if idxOverlappingBoutons is None:
        idxOverlappingBoutons = np.arange(0,positionBoutons_3d.shape[0])
    
    fluoPlaneWidth = positionROI_3d.shape[1]
    fluoPlaneHeight = positionROI_3d.shape[2]
    
    coloredROI = np.zeros((fluoPlaneHeight,fluoPlaneWidth))
    
    # V1 ROIs
    tmp = np.sum(positionROI_3d,axis=0)
    coloredROI = np.clip(coloredROI + tmp,0,2)
    
    # Thalamic boutons
    thoseBoutonsPosition3D = positionBoutons_3d[idxOverlappingBoutons]
    tmp2 = np.clip(2*np.sum(thoseBoutonsPosition3D,axis=0),0,3)
    coloredROI = np.clip(coloredROI + tmp2,0,3)
    
    # Clip the values for overlaps
    coloredROI = np.clip(coloredROI,0,2)
    
    # Create colormap
    cmap = colors.ListedColormap(['white', 'black', 'red'])
    bounds = np.linspace(0,3,4)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure()
    # tell imshow about color map so that only set colors are used
    img = plt.imshow(np.transpose(coloredROI), interpolation='nearest', cmap=cmap, norm=norm)
    
    # make a color bar (it uses 'cmap' and 'norm' form above)
    cbar = plt.colorbar(img, boundaries=np.linspace(1,3,3), ticks=np.linspace(1.5,2.5,2))
    cbar.ax.set_yticklabels(['V1','bouton'])
    
    plt.title('V1 ROIs with thalamic boutons')



# Plot each ROI which has associated boutons with its boutons
def plot_eachROIwithBoutons(positionROI,positionBoutons,fluoPlaneWidth,fluoPlaneHeight,idxOverlappingBoutons,idxAssignedROI):
    
    positionROI = np.array(positionROI)
    positionBoutons = np.array(positionBoutons)
    positionROI_3d = positionROI.reshape((positionROI.shape[0],fluoPlaneWidth,fluoPlaneHeight))
    positionBoutons_3d = positionBoutons.reshape((positionBoutons.shape[0],fluoPlaneWidth,fluoPlaneHeight))
    
    # ROI which have boutons
    theseROI = np.unique(idxAssignedROI)
    
    # Create colormap
    cmap = colors.ListedColormap(['white', 'black', 'red'])
    bounds = np.linspace(0,3,4)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot each ROI with its assigned boutons in a different plot, but maximum 3 of them
    numPlots = np.min((3,theseROI.size))
    if theseROI.size > 3:
        print('We plot only 3 out of the '+ str(theseROI.size) +' V1 ROIs')  
    for i in range(0,numPlots):
        coloredROI = np.zeros((fluoPlaneHeight,fluoPlaneWidth))
        tmpIdxROI = theseROI[i]
        coloredROI = coloredROI + positionROI_3d[tmpIdxROI] # V1 ROI
        tmpIdxBoutons = idxOverlappingBoutons[np.asarray(np.where(idxAssignedROI==tmpIdxROI)).flatten()]
        if tmpIdxBoutons.size>1:
            coloredROI = coloredROI + np.sum(positionBoutons_3d[tmpIdxBoutons],axis=0) # associated thalamic boutons
        else:
            coloredROI = coloredROI + positionBoutons_3d[tmpIdxBoutons] # associated thalamic boutons
        coloredROI = np.clip(coloredROI,0,2) # clip the values for overlaps
        
        plt.figure()
        # tell imshow about color map so that only set colors are used
        img = plt.imshow(np.transpose(coloredROI), interpolation='nearest', cmap=cmap, norm=norm)
        # make a color bar (it uses 'cmap' and 'norm' form above)
        cbar = plt.colorbar(img, boundaries=np.linspace(1,3,3), ticks=np.linspace(1.5,2.5,2))
        cbar.ax.set_yticklabels(['V1','bouton'])
        plt.title('ROI '+str(tmpIdxROI)+' with thalamic boutons')




# Plot the average fluorescence for each trial orientation
def plot_avgFluoPerOri(data,dataFramesPerTrial):
    
    tmpTime = np.arange(dataFramesPerTrial)+1
    avgPerOri = data.drop(['Session','Trial','FrameType'],axis=1).groupby(['TrialFrame','Orientation']).mean()
    avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
    avgPerOri = avgPerOri.reset_index()
    
    fig, axs = plt.subplots(2, 2)
    
    tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[0]]
    tmp = np.array(tmp.drop(['TrialFrame','Orientation'],axis=1))
    tmpMean = np.mean(tmp,axis=1)
    tmpSEM = np.std(tmp,axis=1)/np.sqrt(tmp.shape[1])
    axs[0, 0].plot(tmpTime,tmpMean)
    axs[0, 0].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    axs[0, 0].set_title('Orientation: '+str(globalParams.ori[0])+'°')
    
    tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[1]]
    tmp = np.array(tmp.drop(['TrialFrame','Orientation'],axis=1))
    tmpMean = np.mean(tmp,axis=1)
    tmpSEM = np.std(tmp,axis=1)/np.sqrt(tmp.shape[1])
    axs[0, 1].plot(tmpTime,tmpMean)
    axs[0, 1].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    axs[0, 1].set_title('Orientation: '+str(globalParams.ori[1])+'°')
    
    tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[2]]
    tmp = np.array(tmp.drop(['TrialFrame','Orientation'],axis=1))
    tmpMean = np.mean(tmp,axis=1)
    tmpSEM = np.std(tmp,axis=1)/np.sqrt(tmp.shape[1])
    axs[1, 0].plot(tmpTime,tmpMean)
    axs[1, 0].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    axs[1, 0].set_title('Orientation: '+str(globalParams.ori[2])+'°')
    
    tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[3]]
    tmp = np.array(tmp.drop(['TrialFrame','Orientation'],axis=1))
    tmpMean = np.mean(tmp,axis=1)
    tmpSEM = np.std(tmp,axis=1)/np.sqrt(tmp.shape[1])
    axs[1, 1].plot(tmpTime,tmpMean)
    axs[1, 1].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    axs[1, 1].set_title('Orientation: '+str(globalParams.ori[3])+'°')
    
    
# Plot the average pupil area for each trial orientation
def plot_avgPupilPerOri(data,dataFramesPerTrial):
    
    tmpTime = np.arange(dataFramesPerTrial)+1
    avgPerOri = data.drop(['Session','Trial','FrameType'],axis=1).groupby(['TrialFrame','Orientation']).mean()
    avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
    avgPerOri = avgPerOri.reset_index()
    semPerOri = data.drop(['Session','Trial','FrameType'],axis=1).groupby(['TrialFrame','Orientation']).sem()
    semPerOri = semPerOri.sort_values(['Orientation','TrialFrame'])
    semPerOri = semPerOri.reset_index()
    
    fig, axs = plt.subplots(2, 2)

    tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[0]]
    tmp2 = semPerOri[semPerOri['Orientation']==globalParams.ori[0]]
    tmpMean = np.squeeze(np.array(tmp.drop(['TrialFrame','Orientation'],axis=1)))
    tmpSEM = np.squeeze(np.array(tmp2.drop(['TrialFrame','Orientation'],axis=1)))
    axs[0, 0].plot(tmpTime,tmpMean)
    axs[0, 0].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    axs[0, 0].set_title('Orientation: '+str(globalParams.ori[0])+'°')

    tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[1]]
    tmp2 = semPerOri[semPerOri['Orientation']==globalParams.ori[1]]
    tmpMean = np.squeeze(np.array(tmp.drop(['TrialFrame','Orientation'],axis=1)))
    tmpSEM = np.squeeze(np.array(tmp2.drop(['TrialFrame','Orientation'],axis=1)))
    axs[0, 1].plot(tmpTime,tmpMean)
    axs[0, 1].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    axs[0, 1].set_title('Orientation: '+str(globalParams.ori[1])+'°')

    tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[2]]
    tmp2 = semPerOri[semPerOri['Orientation']==globalParams.ori[2]]
    tmpMean = np.squeeze(np.array(tmp.drop(['TrialFrame','Orientation'],axis=1)))
    tmpSEM = np.squeeze(np.array(tmp2.drop(['TrialFrame','Orientation'],axis=1)))
    axs[1, 0].plot(tmpTime,tmpMean)
    axs[1, 0].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    axs[1, 0].set_title('Orientation: '+str(globalParams.ori[2])+'°')

    tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[3]]
    tmp2 = semPerOri[semPerOri['Orientation']==globalParams.ori[3]]
    tmpMean = np.squeeze(np.array(tmp.drop(['TrialFrame','Orientation'],axis=1)))
    tmpSEM = np.squeeze(np.array(tmp2.drop(['TrialFrame','Orientation'],axis=1)))
    axs[1, 1].plot(tmpTime,tmpMean)
    axs[1, 1].fill_between(tmpTime,tmpMean-tmpSEM,tmpMean+tmpSEM)
    axs[1, 1].set_title('Orientation: '+str(globalParams.ori[3])+'°')
    
    fig.suptitle('Pupil area')


# Preprocess pupil data
def preprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path,charTrials):

    pupilfilepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
            dataMouse + '\\' + dataDepth + '\\pupil_manual.mat'
    
    if not path.isfile(pupilfilepath):
        pupilfilepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
            dataMouse + '\\' + dataDepth + '\\pupil.mat'
    
    if path.isfile(pupilfilepath):
        
        print("Taking care of pupil data...")
        
        # Read pupil data
        # if dataType=='L23_thalamicBoutons':
        f = scipy.io.loadmat(pupilfilepath)
        tmp = f['pupil']
        val = tmp[0,0]
        majorAxisLength = val['MajorAxisLength']
        minorAxisLength = val['MinorAxisLength']
        pupilArea = val['pupilArea']
        xCenter = val['Xc']
        yCenter = val['Yc']
        # elif dataType=='L4_LGN_targeted_axons':
        #     arrays = {}
        #     f = h5py.File(pupilfilepath,'r')
        #     for k, v in f.items():
        #         arrays[k] = np.array(v)
        #     print('Attention!! Still need to write the code for pupil of L4 boutons')

        
        charTrials['majorAxisLength'] = np.transpose(majorAxisLength)
        charTrials['minorAxisLength'] = np.transpose(minorAxisLength)
        charTrials['pupilArea'] = np.transpose(pupilArea)
        charTrials['xCenter'] = np.transpose(xCenter)
        charTrials['yCenter'] = np.transpose(yCenter)
        
    return charTrials


# Preprocess pupil data
def postprocess_pupil(dataType,dataDate,dataMouse,dataDepth,path):

    pupilfilepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
            dataMouse + '\\' + dataDepth + '\\pupil_manual.mat'
    
    if not path.isfile(pupilfilepath):
        pupilfilepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
            dataMouse + '\\' + dataDepth + '\\pupil.mat'
    
    if path.isfile(pupilfilepath):
        
        print("Taking care of pupil data...")
        
        # Read pupil data
        # if dataType=='L23_thalamicBoutons':
        f = scipy.io.loadmat(pupilfilepath)
        tmp = f['pupil']
        val = tmp[0,0]
        majorAxisLength = val['MajorAxisLength']
        minorAxisLength = val['MinorAxisLength']
        pupilArea = val['pupilArea']
        xCenter = val['Xc']
        yCenter = val['Yc']
        # elif dataType=='L4_LGN_targeted_axons':
        #     arrays = {}
        #     f = h5py.File(pupilfilepath,'r')
        #     for k, v in f.items():
        #         arrays[k] = np.array(v)
        #     print('Attention!! Still need to write the code for pupil of L4 boutons')

        charPupil = pd.DataFrame(np.transpose(pupilArea),columns=['pupilArea'])
        charPupil['majorAxisLength'] = np.transpose(majorAxisLength)
        charPupil['minorAxisLength'] = np.transpose(minorAxisLength)
        charPupil['xCenter'] = np.transpose(xCenter)
        charPupil['yCenter'] = np.transpose(yCenter)
        
    else:
        charPupil = None
        
    return charPupil



# Preprocess motion data
def preprocess_motion(dataType,dataDate,dataMouse,dataDepth,path,charTrials):

    motionfilepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
            dataMouse + '\\' + dataDepth + '\\motion.mat'
    
    if path.isfile(motionfilepath):
        
        print("Taking care of motion data...")
        
        f = scipy.io.loadmat(motionfilepath)
        tmp = f['motion']
        val = tmp[0,0]
        motSVD = val['motSVD']
        motAvg = val['mot_avg']
        uMotMask = val['uMotMask']
        
        # Keep only the projection on the first SVD for now
        charTrials['motSVD1'] = motSVD[:,0]
        charTrials['motSVD2'] = motSVD[:,1]
        charTrials['motSVD3'] = motSVD[:,2]
        charTrials['motSVD4'] = motSVD[:,3]
        charTrials['motSVD5'] = motSVD[:,4]
        
        # Plot the average motion
        plt.figure()
        plt.imshow(motAvg)
        
    else:
        motAvg = None
        uMotMask = None
            
        
    return charTrials,motAvg,uMotMask


# Postprocess motion data
def postprocess_motion(dataType,dataDate,dataMouse,dataDepth,path):

    motionfilepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
            dataMouse + '\\' + dataDepth + '\\motion.mat'
    
    if path.isfile(motionfilepath):
        
        print("Taking care of motion data...")
        
        f = scipy.io.loadmat(motionfilepath)
        tmp = f['motion']
        val = tmp[0,0]
        motSVD = pd.DataFrame(val['motSVD'])
        motAvg = val['mot_avg']

        # Plot the average motion
        plt.figure()
        plt.imshow(motAvg)
        
    else:
        motSVD = None
            
    return motSVD



# def loadBoutonDataAll(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub):
    
    
#     this_dff0 = []
#     this_order = []
#     this_baselineRaw = []
#     this_positionROI3d = []
#     nTrialsPerSession = np.zeros(len(dataSessions),dtype=int)
    
#     for filePartNum in range(0,4): # data is cut in 4 pieces
    
#         for s in range(len(dataSessions)):
        
#             sessionNumber = dataSessions[s]
            
#             filepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
#                         dataMouse + '\\' + dataDepth + '\\ThalamicAxons\\S' + str(sessionNumber) + \
#                             '\\piece' + str(filePartNum+1) + '\\ROI_' + dataDate + \
#                                 '_RM_piece' + str(filePartNum+1) + '_S' + str(sessionNumber) + \
#                                     '_Intensity_unweighted_s2p_' + dataNeuropilSub + '.mat'
            
#             arrays = {}
#             f = h5py.File(filepath,'r')
#             for k, v in f.items():
#                 arrays[k] = np.array(v)
                
                
#             this_dff0 = arrays['dff0'] # fluorescence data
#             this_order = arrays['order'][0] # the orientation for each frame (also blank frames)
#             this_baselineRaw = arrays['baseline_raw']
#             this_positionROI3d = arrays['bw']
                
#             if filePartNum==0:
#                 nTrialsPerSession[s] = int(this_dff0.shape[1]/30) # 30 frames/trial
            
#             # Concatenate the data
#             if s==0:
#                 dff0 = this_dff0
#                 order = this_order
#             else:
#                 dff0 = np.concatenate((dff0,this_dff0),axis=1)
#                 order = np.concatenate((order,this_order),axis=0)
                
#             # Save info about the ROIs for the first session only
#             if s==0:
#                 if filePartNum==0:
#                     baseline_raw_piece1 = this_baselineRaw
#                     positionROI_3d_piece1 = this_positionROI3d
#                 elif filePartNum==1:
#                     baseline_raw_piece2 = this_baselineRaw
#                     positionROI_3d_piece2 = this_positionROI3d
#                 elif filePartNum==2:
#                     baseline_raw_piece3 = this_baselineRaw
#                     positionROI_3d_piece3 = this_positionROI3d
#                 elif filePartNum==3:
#                     baseline_raw_piece4 = this_baselineRaw
#                     positionROI_3d_piece4 = this_positionROI3d
            
#             if filePartNum==0:
#                 dff0_piece1 = dff0
#                 order_piece1 = order
#             elif filePartNum==1:
#                 dff0_piece2 = dff0
#                 order_piece2 = order
#             elif filePartNum==2:
#                 dff0_piece3 = dff0
#                 order_piece3 = order
#             elif filePartNum==3:
#                 dff0_piece4 = dff0
#                 order_piece4 = order
                    
    
#     return nTrialsPerSession, \
#         dff0_piece1,dff0_piece2,dff0_piece3,dff0_piece4, \
#             order_piece1,order_piece2,order_piece3,order_piece4, \
#                 baseline_raw_piece1,baseline_raw_piece2,baseline_raw_piece3,baseline_raw_piece4, \
#                     positionROI_3d_piece1,positionROI_3d_piece2,positionROI_3d_piece3,positionROI_3d_piece4



def loadBoutonData(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub,nBlankFrames,nStimFrames):
    
    nTrialsPerSession = np.zeros(len(dataSessions),dtype=int)
    dataFramesPerTrial = nBlankFrames + nStimFrames
    
    for s in range(len(dataSessions)):
        
        print('Loading session '+ str(s+1) +' out of '+str(len(dataSessions)))  
        
        sessionNumber = dataSessions[s]
        
        if dataType=='L23_thalamicBoutons':
            filepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
                        dataMouse + '\\' + dataDepth + '\\Final_Revised_Boutons\\ROI_' + dataDate + \
                                '_' + dataDepth + '_S' + str(sessionNumber) + \
                                    '_Intensity_unweighted_s2p_' + dataNeuropilSub + '.mat'
        elif dataType=='L4_LGN_targeted_axons':
            filepath = globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
                        dataMouse + '\\' + dataDepth + '\\ROI_' + dataDate + \
                                '_S' + str(sessionNumber) + \
                                    '_Intensity_unweighted_s2p_' + dataNeuropilSub + '.mat'
        
        arrays = {}
        f = h5py.File(filepath,'r')
        for k, v in f.items():
            arrays[k] = np.array(v)
            
            
        this_dff0 = arrays['dff0_allstacks'] # fluorescence data
        this_order = arrays['order'][0] # the orientation for each frame (also blank frames)
        this_baselineRaw = np.mean(arrays['baseline_vector_allstacks'],axis=1) # Baseline is different for each trial, so we average the baseline over all trials
        this_positionROI3d = arrays['bw']
            
        nTrialsPerSession[s] = int(this_dff0.shape[1]/dataFramesPerTrial)
        
        # Concatenate the data
        if s==0:
            dff0 = this_dff0
            order = this_order
            baseline_raw = this_baselineRaw[:,np.newaxis]
            positionROI_3d = this_positionROI3d  # The ROI positions do not change over sessions
        else:
            dff0 = np.concatenate((dff0,this_dff0),axis=1)
            order = np.concatenate((order,this_order),axis=0)
            baseline_raw = np.concatenate((baseline_raw,this_baselineRaw[:,np.newaxis]),axis=1)
    
    # Compute the average baseline over all recording sessions
    baseline_raw = np.mean(baseline_raw,axis=1)                  
            
    return nTrialsPerSession,dff0,order,baseline_raw,positionROI_3d




def selectBoutonROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,order,dff0,positionROI_3d,nBlankFrames,nStimFrames):
    
    # Parameters
    nROI_init = dff0.shape[0] # number of boutons before preprocessing
    nFrames = dff0.shape[1] # number of fluorescence frames
    dataFramesPerTrial = nBlankFrames + nStimFrames
    nTrials = int(nFrames/dataFramesPerTrial)
    fluoPlaneWidth = positionROI_3d.shape[1]
    fluoPlaneHeight = positionROI_3d.shape[2]
    
    # Create a DataFrame for the trial characteristics
    charTrials = pd.DataFrame(np.repeat(dataSessions,dataFramesPerTrial*(nTrialsPerSession)),columns=['Session'])
    charTrials['Trial'] = np.repeat(np.arange(nTrials),dataFramesPerTrial)
    charTrials['TrialFrame'] = np.tile(np.arange(dataFramesPerTrial),nTrials)
    tmp = np.concatenate((np.repeat(['Blank'],nBlankFrames),\
                          np.repeat(['Stimulus'],nStimFrames)))
    frameType = np.tile(tmp,nTrials)
    charTrials['FrameType'] = frameType
    charTrials['Orientation'] = order
    
    # (1) Set 'inf' dff0 values to 'nan'
    tmpBoolInf = np.isinf(dff0)
    dff0[tmpBoolInf] = np.nan
    
    # (2) Set dff0 values > 1000% or < 1000% to 'nan'
    tmpBoolTooHigh = (dff0>1000)
    tmpBoolTooLow = (dff0<-1000)
    dff0[tmpBoolTooHigh] = np.nan
    dff0[tmpBoolTooLow] = np.nan
    
    # (3) Discard boutons that have more than 25% of dff0 values which are 'nan'
    tmpBoolNan = np.isnan(dff0)
    tmpSum = np.sum(tmpBoolNan,axis=1)
    tmpIdx = np.where((tmpSum/nFrames) < 0.25)[0]
    dff0 = dff0[tmpIdx,:]
    positionROI_3d = positionROI_3d[tmpIdx]
    
    # Number of ROIs we keep after postprocessing
    nROI = dff0.shape[0]
    print('We keep '+str(nROI)+' boutons out of '+str(nROI_init)+' recorded boutons ('+str(np.round(100*nROI/nROI_init))+'%).')
    
    # # (2) Discard ROIs with a value >1500 for L4 and >2000 for L2/3
    # tmpMax = np.max(dff0,axis=1)
    # if dataType=='L23_thalamicBoutons':
    #     tmpIdx = np.where(tmpMax<25000)[0] # np.where(tmpMax<2000)[0]
    # elif dataType=='L4_LGN_targeted_axons':
    #     tmpIdx = np.where(tmpMax<1500)[0]
    # baseline_raw = baseline_raw[tmpIdx]
    # dff0 = dff0[tmpIdx,:]
    # positionROI_3d = positionROI_3d[tmpIdx]
    
    # Create a DataFrame for the ROI positions
    positionROI = pd.DataFrame(positionROI_3d.reshape(positionROI_3d.shape[0],fluoPlaneWidth*fluoPlaneHeight))
    
    # Create a DataFrame for the fluorescence data
    fluo_array = np.transpose(dff0)
    fluo = pd.DataFrame(np.transpose(dff0))
    
    # Create a DataFrame for the ROI characteristics
    charROI = pd.DataFrame()
    
    data = pd.concat([charTrials,fluo],axis=1)
    
    ### start 1
    # Average fluorescence value for blank frames and stimulus frames for each trial
    avgFluo_Blank = data.loc[data['FrameType']=='Blank'].groupby(['Orientation','Trial']).mean()
    avgFluo_Blank = avgFluo_Blank.droplevel(level=1)
    avgFluo_Blank = avgFluo_Blank.drop(columns=['Session','TrialFrame'])
    avgFluo_Stimulus = data.loc[data['FrameType']=='Stimulus'].groupby(['Orientation','Trial']).mean()
    avgFluo_Stimulus = avgFluo_Stimulus.droplevel(level=1)
    avgFluo_Stimulus = avgFluo_Stimulus.drop(columns=['Session','TrialFrame'])
    
    # Anova test to determine which ROIs are visually responsive (=are activated by the grating stimuli)
    bool_visuallyResp = np.zeros((nROI,globalParams.nOri))
    for o in range(globalParams.nOri):
        thisB = avgFluo_Blank.loc[globalParams.ori[o]]
        thisG = avgFluo_Stimulus.loc[globalParams.ori[o]]
        F,pval = stats.f_oneway(thisB,thisG)
        bool_visuallyResp[:,o] = (pval < globalParams.threshold_pval_visuallyEvoked)
    
    idxVE = np.where(np.sum(bool_visuallyResp,axis=1) > 0)[0]
    charROI['VisuallyEvoked'] =  np.array([True if i in idxVE else False for i in range(nROI)])
    
    # Orientation preference
    tmpAvg = avgFluo_Stimulus.groupby('Orientation').mean()
    charROI['PrefOri'] = tmpAvg.idxmax(axis=0)
    
    print(str(len(np.where(charROI['VisuallyEvoked']==True)[0]))+' out of '+str(nROI)+' ROIs are visually responsive ('+str(np.round(100*len(np.where(charROI['VisuallyEvoked']==True)[0])/nROI).astype(int))+'%)')
    ### end 1
    
    # Size of ROIs
    charROI['Size'] = positionROI.sum(axis=1)
    
    # Center of mass of ROIs (scipy.ndimage.center_of_mass)
    for n in range(dff0.shape[0]):
        this_xCM, this_yCM = center_of_mass(positionROI_3d[n])
        this_xCM = int(np.round(this_xCM))
        this_yCM = int(np.round(this_yCM))
        if n==0:
            cmROI = np.array([[this_xCM,this_yCM]])
        else:
            cmROI = np.append(cmROI,np.array([[this_xCM,this_yCM]]),axis=0)
            
    charROI['xCM'] = pd.DataFrame(cmROI[:,0])
    charROI['yCM'] = pd.DataFrame(cmROI[:,1])
    
    # Compute pairwise distance between all ROIs
    distROI = pixelSize*euclidean_distances(cmROI)
    
    ### start 2
    # Anova test to determine orientation selectivity
    group0 = avgFluo_Stimulus.loc[globalParams.ori[0]]
    group1 = avgFluo_Stimulus.loc[globalParams.ori[1]]
    group2 = avgFluo_Stimulus.loc[globalParams.ori[2]]
    group3 = avgFluo_Stimulus.loc[globalParams.ori[3]]
    f_val, p_val = stats.f_oneway(group0,group1,group2,group3)
    bool_OS = (p_val < globalParams.threshold_pval_OS)
    
    # Not visually evoked ROIs cannot be orientation selective
    idxNotVE = np.where((charROI['VisuallyEvoked']==False)&(bool_OS==True))[0]
    bool_OS[idxNotVE] = 0
    
    # Are ROIs orientation-selective
    nOS = len(np.where(bool_OS==True)[0])
    nVE = len(np.where(charROI['VisuallyEvoked']==True)[0])
    print('There are '+str(nOS)+' visually evoked ROIs which are orientation selective ('+str(np.round(100*nOS/nVE).astype(int))+'%)')
    charROI['OS'] = np.array([True if bool_OS[i]==1 else False for i in range(nROI)])  
    print(str(nOS)+' boutons are OS ('+str(np.round(100*nOS/nROI))+'%).')
    ### end 2
    
    # # Average value over all stimulus frames for each ROI and each trial of each orientation
    # avgFluoPerTrial = data.loc[data['FrameType']=='Stimulus'].groupby('Trial').mean()
    # avgFluoPerTrial = avgFluoPerTrial.drop(columns=['Session','TrialFrame']) # no need for session number and trial frame number
    # avgFluoPerTrial = avgFluoPerTrial.reset_index(drop=True).set_index('Orientation').sort_index()
    
    # # Anova test to determine orientation selectivity
    # bool_OS = np.zeros(nROI)
    # for n in range(nROI):
    #     this_data = avgFluoPerTrial[n]
        
    #     group0 = this_data[this_data.index==globalParams.ori[0]]
    #     group1 = this_data[this_data.index==globalParams.ori[1]]
    #     group2 = this_data[this_data.index==globalParams.ori[2]]
    #     group3 = this_data[this_data.index==globalParams.ori[3]]
    
    #     f_val, p_val = stats.f_oneway(group0,group1,group2,group3)
    #     if p_val < globalParams.threshold_pval:
    #         bool_OS[n] = 1
    
    # # (5) Are ROIs orientation-selective? (and which orientation are they most responsive to?)
    # charROI['OS'] = np.array([True if bool_OS[i]==1 else False for i in range(nROI)])
    
    
    ## Plotting after getting rid of bad ROIs ##

    # Plot all kept ROIs at the end of postprocessing
    plt.figure()
    plt.imshow(np.transpose(np.sum(positionROI_3d,axis=0)))
    plt.title('Thalamic boutons kept after postprocessing')
    
    # Plot all kept ROIs with their selectivity
    plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)
    
    # Plot the average fluorescence for each trial orientation
    plot_avgFluoPerOri(data,dataFramesPerTrial)
    
    return charROI,charTrials,fluo,fluo_array,data,positionROI,distROI



# Get the indices of the overlapping boutons (and the V1 ROI they are assigned to)
def getOverlappingBoutonROI(positionROI,positionBoutons):
    
    nROI = positionROI.shape[0]
    nBoutons = positionBoutons.shape[0]
    
    idxAllROIs = np.asarray(np.where(np.array(positionROI.sum())>0)).flatten()
    
    idxOverlappingBoutons = []
    idxAssignedROI = []
    for b in range(0,nBoutons):
        tmpIdxBouton = np.asarray(np.where(np.array(positionBoutons.loc[b])>0)).flatten()
        if np.intersect1d(tmpIdxBouton,idxAllROIs).size > 0:
            idxOverlappingBoutons = np.append(idxOverlappingBoutons,b) # this bouton connects to a V1 ROI
            for r in range(0,nROI):
                tmpIdxROI = np.asarray(np.where(np.array(positionROI.loc[r])>0)).flatten()
                if np.intersect1d(tmpIdxBouton,tmpIdxROI).size > 0:
                    idxAssignedROI = np.append(idxAssignedROI,r)
                    break
    
    idxOverlappingBoutons = idxOverlappingBoutons.astype(int)
    idxAssignedROI = idxAssignedROI.astype(int)
    
    return idxAssignedROI,idxOverlappingBoutons







# def loadData(dataType,dataSessions,dataDate,dataMouse,dataDepth,dataNeuropilSub,nBlankFrames,nStimFrames):
    
#     nTrialsPerSession = np.zeros(len(dataSessions),dtype=int)
#     for s in range(len(dataSessions)):
    
#         sessionNumber = dataSessions[s]
#         dataFramesPerTrial = nBlankFrames + nStimFrames
        
#         filepath = []
#         if dataType == 'L4_cytosolic':
#             filepath.append(globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
#                 dataMouse + '\\' + dataDepth + '\\S' + str(sessionNumber) + \
#                     '\\ROI_' + dataDate + '_RM_S' + str(sessionNumber) + \
#                         '_Intensity_unweighted_s2p_' + dataNeuropilSub + '.mat')
#             # filepath.append(globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
#             #     dataMouse + '\\' + dataDepth + '\\Updated_July2022\\' + \
#             #         '\\ROI_' + dataDate + '_S' + str(sessionNumber) + \
#             #             '_Intensity_unweighted_imj_' + dataNeuropilSub + '.mat')
                
#         elif dataType == 'L23_thalamicBoutons':
#             filepath.append(globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
#                                 dataMouse + '\\' + dataDepth + '\\L23\\ROI_' + \
#                                     '' + dataDate + '_' + dataDepth + '_S' + str(sessionNumber) + \
#                                             '_Intensity_unweighted_' + dataNeuropilSub + '.mat')
        
        
#         this_dff0 = []
#         this_order = []
#         this_baselineRaw = []
#         this_positionROI3d = []

#         arrays = {}
#         f = h5py.File(filepath[0],'r')
#         for k, v in f.items():
#             arrays[k] = np.array(v)
        
        
#         this_order = arrays['order'][0] # the orientation for each frame (also blank frames)
#         if dataType == 'L4_cytosolic':
#             this_dff0 = arrays['dff0'] # fluorescence data
#             this_baselineRaw = arrays['baseline_raw']
#         elif dataType == 'L23_thalamicBoutons':
#             this_dff0 = arrays['dff0_allstacks'] # fluorescence data
#             this_baselineRaw = arrays['baseline_vector_allstacks'][:,0]
#         nTrialsPerSession[s] = int(this_dff0.shape[1]/dataFramesPerTrial)
#         this_positionROI3d = arrays['bw']
        
#         # Concatenate the data
#         if s==0:
#             dff0 = this_dff0
#             order = this_order
#             baseline_raw = this_baselineRaw[:,np.newaxis]
#         else:
#             dff0 = np.concatenate((dff0,this_dff0),axis=1)
#             order = np.concatenate((order,this_order),axis=0)
#             baseline_raw = np.concatenate((baseline_raw,this_baselineRaw[:,np.newaxis]),axis=1)
            
#         # Save info about the ROIs for the first session only
#         if s==0:
#             positionROI_3d = this_positionROI3d
    
#     # Compute the average baseline over all recording sessions
#     baseline_raw = np.mean(baseline_raw,axis=1)
    
#     return dff0,order,baseline_raw,positionROI_3d,nTrialsPerSession



# def selectROIs(dataType,pixelSize,dataSessions,nTrialsPerSession,baseline_raw,order,dff0,positionROI_3d,nBlankFrames,nStimFrames):
    
#     # Parameters
#     nROI_init = dff0.shape[0] # number of neurons before preprocessing
#     nFrames = dff0.shape[1] # number of fluorescence frames
#     dataFramesPerTrial = nBlankFrames + nStimFrames
#     nTrials = int(nFrames/dataFramesPerTrial)
#     fluoPlaneWidth = positionROI_3d.shape[1]
#     fluoPlaneHeight = positionROI_3d.shape[2]
    
#     # Create a DataFrame for the ROI positions
#     positionROI = pd.DataFrame(positionROI_3d.reshape(nROI_init,fluoPlaneWidth*fluoPlaneHeight))
    
#     # Create a DataFrame for the fluorescence data
#     fluo_array = np.transpose(dff0)
#     fluo = pd.DataFrame(np.transpose(dff0))

#     # Create a DataFrame for the ROI characteristics
#     charROI = pd.DataFrame(baseline_raw,columns=['BaselineRaw'])
    
#     # Create a DataFrame for the trial characteristics
#     charTrials = pd.DataFrame(np.repeat(dataSessions,dataFramesPerTrial*(nTrialsPerSession)),columns=['Session'])
#     charTrials['Trial'] = np.repeat(np.arange(nTrials),dataFramesPerTrial)
#     charTrials['TrialFrame'] = np.tile(np.arange(dataFramesPerTrial),nTrials)
    
#     tmp = np.concatenate((np.repeat(['Blank'],nBlankFrames),\
#                           np.repeat(['Stimulus'],nStimFrames))) 
#     frameType = np.tile(tmp,nTrials)
#     charTrials['FrameType'] = frameType
#     charTrials['Orientation'] = order
    
    
#     data = pd.concat([charTrials,fluo],axis=1)
    
#     # Select ROIs which respond to grating stimuli
#     #avgFluo = data.loc[data['FrameType']=='Stimulus'].groupby('Orientation').mean()
#     avgFluo = data.groupby(['Orientation','TrialFrame']).mean()
    
#     # Orientation preference
#     charROI['MaxDff0'] = avgFluo.max(axis=0)
#     #charROI['PrefOri'] = avgFluo.idxmax(axis=0)
#     charROI['PrefOri'] = avgFluo.droplevel(level=1).idxmax(axis=0)
    
#     print(str(len(np.where(charROI['MaxDff0']>10)[0]))+' out of '+str(charROI.shape[0])+' ROIs have MaxDff0 > 10%')
    
#     # Select ROIs which are sufficiently active
#     percBaselineRaw = charROI['BaselineRaw'].quantile(q=globalParams.threshold_percentile)
    
#     # Size of ROIs (unit: pixels)
#     charROI['Size'] = positionROI.sum(axis=1)
    
#     # Center of mass of ROIs (scipy.ndimage.center_of_mass)
#     for n in range(nROI_init):
#         this_xCM, this_yCM = center_of_mass(positionROI_3d[n])
#         this_xCM = int(np.round(this_xCM))
#         this_yCM = int(np.round(this_yCM))
#         if n==0:
#             cmROI = np.array([[this_xCM,this_yCM]])
#         else:
#             cmROI = np.append(cmROI,np.array([[this_xCM,this_yCM]]),axis=0)
            
    
#     charROI['xCM'] = pd.DataFrame(cmROI[:,0])
#     charROI['yCM'] = pd.DataFrame(cmROI[:,1])
    
    
#     # Compute pairwise distance between all ROIs
#     distROI = pixelSize*euclidean_distances(cmROI)
    
#     # Find pairs of ROIs which are too close to each other
#     tmp_idxROItooClose = np.argwhere(distROI<globalParams.threshold_distance)
    
#     # Find the ROI which are the smallest of a pair which is too close (to discard)
#     idxROItooClose = []
#     for i in range(tmp_idxROItooClose.shape[0]):
#         this_pair = tmp_idxROItooClose[i,:]
#         if this_pair[0] != this_pair[1]:
#             if ~np.isin(this_pair[0],idxROItooClose) & ~np.isin(this_pair[1],idxROItooClose):
#                 tmpSize0 = charROI['Size'][this_pair[0]]
#                 tmpSize1 = charROI['Size'][this_pair[1]]
#                 if tmpSize0 < tmpSize1:
#                     idxROItooClose = np.append(idxROItooClose,this_pair[0])
#                 else:
#                     idxROItooClose = np.append(idxROItooClose,this_pair[1])
    
#     # (3) Are ROIs far enough? (if not: discard smallest one of the pair)
#     charROI['farEnough'] = np.array([False if i in idxROItooClose else True for i in range(nROI_init)])
    
#     # (3bis) Are ROIs big enough (in terms of number of pixels)?
#     if dataType == 'L23_thalamicBoutons': # NB: before threshold was set to 1000
#         charROI['bigEnough'] = np.array([False if np.array(charROI['Size'])[i]<500 else True for i in range(nROI_init)])
    
#     # # (4) Set all negative df/f0 values to zero
#     # data.loc[:,~data.columns.isin(['Session','Trial','TrialFrame','FrameType','Orientation'])] = data.drop(['Session','Trial','TrialFrame','FrameType','Orientation'],axis=1).clip(lower=0)
#     # fluo_array = np.clip(fluo_array,a_min=0,a_max=None)
    
#     # Average value over all stimulus frames for each ROI and each trial of each orientation
#     avgFluoPerTrial = data.loc[data['FrameType']=='Stimulus'].groupby('Trial').mean()
#     avgFluoPerTrial = avgFluoPerTrial.drop(columns=['Session','TrialFrame']) # no need for session number and trial frame number
#     avgFluoPerTrial = avgFluoPerTrial.reset_index(drop=True).set_index('Orientation').sort_index()
    
#     # Anova test to determine orientation selectivity
#     bool_OS = np.zeros(nROI_init)
#     for n in range(nROI_init):
#         this_data = avgFluoPerTrial[n]
        
#         group0 = this_data[this_data.index==globalParams.ori[0]]
#         group1 = this_data[this_data.index==globalParams.ori[1]]
#         group2 = this_data[this_data.index==globalParams.ori[2]]
#         group3 = this_data[this_data.index==globalParams.ori[3]]
    
#         f_val, p_val = stats.f_oneway(group0,group1,group2,group3)
#         if p_val < globalParams.threshold_pval:
#             bool_OS[n] = 1
    
#     # (5) Are ROIs orientation-selective? (and which orientation are they most responsive to?)
#     charROI['OS'] = np.array([True if bool_OS[i]==1 else False for i in range(nROI_init)])
    
    
#     # (1) Are ROIs active enough?
#     charROI['activeEnough'] = charROI['BaselineRaw']>percBaselineRaw
    
#     # (2) Are ROIs responsive enough?
#     if dataType=='L4_cytosolic':
#         charROI['responsiveEnough'] = charROI['MaxDff0']>globalParams.threshold_dff0
#     elif dataType=='L23_thalamicBoutons':
#         charROI['responsiveEnough'] = charROI['MaxDff0']>1 #1.0
    
#     # Determine which ROIs we keep for the analyses
#     if dataType=='L4_cytosolic':
#         charROI['keptROI'] = charROI['activeEnough'] & charROI['responsiveEnough'] & charROI['farEnough']
#     elif dataType=='L23_thalamicBoutons':
#         charROI['keptROI'] = charROI['activeEnough'] & charROI['responsiveEnough'] & charROI['farEnough'] & charROI['bigEnough']
    
    
#     ## Plotting before getting rid of bad ROIs ##
    
#     # Plot a random ROI and its center of mass in red
#     plt.figure()
#     rndROI = rnd.randint(0,nROI_init-1)
#     plt.imshow(np.transpose(positionROI_3d[rndROI]))
#     plt.scatter(cmROI[rndROI,0],cmROI[rndROI,1],color='r',s=1)
#     plt.title('Random ROI with CM')
    
#     # Plot all initial ROI
#     plt.figure()
#     plt.imshow(np.transpose(np.sum(positionROI_3d,axis=0)))
#     plt.title('All original ROIs')
    
#     idxNotActEnough = np.where(charROI['activeEnough']==False)[0]
#     plt.figure()
#     plt.imshow(np.transpose(np.sum(positionROI_3d[idxNotActEnough],axis=0)))
#     plt.title('Not active enough ROIs')
    
#     idxNotRespEnough = np.where(charROI['responsiveEnough']==False)[0]
#     plt.figure()
#     plt.imshow(np.transpose(np.sum(positionROI_3d[idxNotRespEnough],axis=0)))
#     plt.title('Not responsive enough ROIs')
    
    
#     # Discard bad ROIs
#     idxKept = np.squeeze(np.array(np.where(np.array(charROI['keptROI']))))
#     fluo_array = fluo_array[:,idxKept]
#     fluo = pd.DataFrame(fluo_array)
#     positionROI = pd.DataFrame(np.array(positionROI)[idxKept,:])
#     positionROI_3d = positionROI_3d[idxKept,:,:]
#     distROI = distROI[idxKept,:][:,idxKept]
#     charROI = charROI[charROI['keptROI']==True].reset_index()
#     if dataType=='L4_cytosolic':
#         charROI = charROI.drop(columns=['index','farEnough','activeEnough','responsiveEnough','keptROI'])
#     elif dataType=='L23_thalamicBoutons':
#         charROI = charROI.drop(columns=['index','farEnough','bigEnough','activeEnough','responsiveEnough','keptROI'])
#     data = pd.concat([charTrials, fluo],axis=1)
    
#     # Number of ROIs we keep after preprocessing
#     nROI = charROI.shape[0]
    
#     ## Plotting after getting rid of bad ROIs ##

#     # Plot all kept ROIs at the end of preprocessing
#     plt.figure()
#     plt.imshow(np.transpose(np.sum(positionROI_3d,axis=0)))
#     plt.title('ROIs kept after preprocessing')
    
#     # Plot all kept ROIs with their selectivity
#     plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)
    
#     # Plot the average fluorescence for each trial orientation
#     plot_avgFluoPerOri(data,dataFramesPerTrial)
    
#     print('We keep '+str(nROI)+' ROIs out of '+str(nROI_init)+' recorded ROIs ('+str(np.round(100*nROI/nROI_init))+'%).')
#     nOS = len(np.where(charROI['OS']==True)[0])
#     print(str(nOS)+' ROIs are OS ('+str(np.round(100*nOS/nROI))+'%).')
        
        
#     return charROI,charTrials,fluo,fluo_array,data,percBaselineRaw,positionROI,distROI,idxKept
