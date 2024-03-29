# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:23:57 2021

@author: Olivia Gozel

Functions to do the preprocessing
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.io
from scipy import ndimage
import scipy.stats as stats
from sklearn.metrics import pairwise

import os
os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams




def loadData(dataType, dataSessions, dataDate, dataMouse, dataDepth, 
             dataNeuropilSub, nBlankFrames, nStimFrames):
    """Load all sessions of one recording dataset."""
    
    # Initialization
    nTrialsPerSession = np.zeros(len(dataSessions), dtype=int)
    
    # Loop over all sessions of the recording dataset of interest
    for s in range(len(dataSessions)):
    
        sessionNumber = dataSessions[s]
        dataFramesPerTrial = nBlankFrames + nStimFrames
        
        filepath = []
        if dataType == 'L4_cytosolic':
            filepath.append(globalParams.dataDir + dataType +'\\' + dataDate \
                            + '_' + dataMouse + '\\' + dataDepth + '\\S' \
                            + str(sessionNumber) + '\\ROI_' + dataDate \
                            + '_RM_S' + str(sessionNumber) \
                            + '_Intensity_unweighted_s2p_' + dataNeuropilSub \
                            + '.mat')
            # filepath.append(globalParams.dataDir + dataType +'\\' + dataDate + '_' + \
            #     dataMouse + '\\' + dataDepth + '\\Updated_July2022\\' + \
            #         '\\ROI_' + dataDate + '_S' + str(sessionNumber) + \
            #             '_Intensity_unweighted_imj_' + dataNeuropilSub + '.mat')
                
        elif dataType == 'L23_thalamicBoutons':
            filepath.append(globalParams.dataDir + dataType +'\\' + dataDate \
                            + '_' + dataMouse + '\\' + dataDepth \
                            + '\\L23\\ROI_' + '' + dataDate + '_' + dataDepth \
                            + '_S' + str(sessionNumber) \
                            + '_Intensity_unweighted_' + dataNeuropilSub \
                            + '.mat')
        
        # Make sure that the following variables are empty
        this_dff0 = []
        this_order = []
        this_positionROI3d = []
        
        # Read current recording session
        arrays = {}
        f = h5py.File(filepath[0], 'r')
        for k, v in f.items():
            arrays[k] = np.array(v)
        
        # Grating orientation for each frame (also blank frames)
        this_order = arrays['order'][0]
        # Fluorescence data
        if dataType == 'L4_cytosolic':
            this_dff0 = arrays['dff0']
        elif dataType == 'L23_thalamicBoutons':
            this_dff0 = arrays['dff0_allstacks']
        nTrialsPerSession[s] = int(this_dff0.shape[1]/dataFramesPerTrial)
        this_positionROI3d = arrays['bw']
        
        # Concatenate the data
        if s==0:
            dff0 = this_dff0
            order = this_order
        else:
            dff0 = np.concatenate((dff0, this_dff0), axis=1)
            order = np.concatenate((order, this_order), axis=0)
            
        # Save info about the ROIs for the first session only
        if s==0:
            positionROI_3d = this_positionROI3d
    
    return dff0, order, positionROI_3d, nTrialsPerSession


def determine_params(dff0, positionROI_3d, nBlankFrames, nStimFrames):
    """ Determine the parameters of the current recording dataset, once all
    sessions are combined."""
    
    # Number of neurons before postprocessing
    nROI_init = dff0.shape[0]
    # Total number of fluorescence frames
    nFrames = dff0.shape[1]
    # Number of frames per trial
    dataFramesPerTrial = nBlankFrames + nStimFrames
    # Total number of trials
    nTrials = int(nFrames/dataFramesPerTrial)
    # Width and height of the imaging plane
    fluoPlaneWidth = positionROI_3d.shape[1]
    fluoPlaneHeight = positionROI_3d.shape[2]
    
    return (nROI_init, nTrials, fluoPlaneWidth, fluoPlaneHeight, nFrames, 
            dataFramesPerTrial)


def selectROIs(dataType, pixelSize, dataSessions, nTrialsPerSession, order, 
               dff0, positionROI_3d, nBlankFrames, nStimFrames):
    """Select ROIs to be kept for subsequent analysis, and classify them into 
    different categories."""
    
    # Parameters for this dataset
    (nROI_init, nTrials, fluoPlaneWidth, fluoPlaneHeight, nFrames, 
     dataFramesPerTrial) = determine_params(dff0, positionROI_3d, nBlankFrames,
                                            nStimFrames)
    
    # (1) Set 'inf' dff0 values to 'nan'
    tmpBoolInf = np.isinf(dff0)
    dff0[tmpBoolInf] = np.nan
    
    # (2) Set dff0 values > 1000% or < 1000% to 'nan'
    tmpBoolTooHigh = (dff0>1000)
    tmpBoolTooLow = (dff0<-1000)
    dff0[tmpBoolTooHigh] = np.nan
    dff0[tmpBoolTooLow] = np.nan
    
    # (3) Discard boutons that have >=25% of dff0 values which are 'nan'
    tmpBoolNan = np.isnan(dff0)
    tmpSum = np.sum(tmpBoolNan, axis=1)
    tmpIdx = np.where((tmpSum/nFrames) < 0.25)[0]
    dff0 = dff0[tmpIdx,:]
    positionROI_3d = positionROI_3d[tmpIdx]
    
    # Number of ROIs we keep
    nROI = dff0.shape[0]
    print('We intially keep ' + str(nROI) + ' ROIs out of ' + str(nROI_init) \
          +' recorded ROIs (' + str( np.round(100*nROI/nROI_init) ) + '%).')
    nROI_init = nROI
    
    # Create a DataFrame for the ROI positions
    positionROI = pd.DataFrame(
        positionROI_3d.reshape(nROI_init, fluoPlaneWidth*fluoPlaneHeight)
        )
    
    # Create a DataFrame for the fluorescence data
    fluo_array = np.transpose(dff0)
    fluo = pd.DataFrame(np.transpose(dff0))

    # Create a DataFrame for the ROI characteristics
    charROI = pd.DataFrame()
    
    # Create a DataFrame for the trial characteristics
    charTrials = pd.DataFrame(
        np.repeat(dataSessions, dataFramesPerTrial*(nTrialsPerSession)), 
        columns=['Session']
        )
    charTrials['Trial'] = np.repeat(np.arange(nTrials), dataFramesPerTrial)
    charTrials['TrialFrame'] = np.tile(np.arange(dataFramesPerTrial), nTrials)
    
    tmp = np.concatenate((np.repeat(['Blank'], nBlankFrames),
                          np.repeat(['Stimulus'], nStimFrames))) 
    frameType = np.tile(tmp, nTrials)
    charTrials['FrameType'] = frameType
    charTrials['Orientation'] = order
    
    data = pd.concat([charTrials, fluo], axis=1)
    
    # Average fluorescence value for blank and stimulus frames for each trial
    tmpBlank = data.loc[data['FrameType']=='Blank']
    avgFluo_Blank = tmpBlank.groupby(['Orientation','Trial']).mean()
    avgFluo_Blank = avgFluo_Blank.droplevel(level=1)
    avgFluo_Blank = avgFluo_Blank.drop(columns=['Session', 'TrialFrame'])
    tmpStim = data.loc[data['FrameType']=='Stimulus']
    avgFluo_Stimulus = tmpStim.groupby(['Orientation','Trial']).mean()
    avgFluo_Stimulus = avgFluo_Stimulus.droplevel(level=1)
    avgFluo_Stimulus = avgFluo_Stimulus.drop(columns=['Session', 'TrialFrame'])
    
    # Anova test to determine which ROIs are visually responsive
    # (=are activated by the grating stimuli)
    bool_visuallyResp = np.zeros((nROI_init, globalParams.nOri))
    for o in range(globalParams.nOri):
        thisB = avgFluo_Blank.loc[globalParams.ori[o]]
        thisG = avgFluo_Stimulus.loc[globalParams.ori[o]]
        F, pval = stats.f_oneway(thisB, thisG)
        bool_visuallyResp[:,o] = \
            (pval < globalParams.threshold_pval_visuallyEvoked)
    
    idxVE = np.where(np.sum(bool_visuallyResp, axis=1) > 0)[0]
    charROI['VisuallyEvoked'] = \
        np.array([True if i in idxVE else False for i in range(nROI_init)])
    
    # Orientation preference
    tmpAvg = avgFluo_Stimulus.groupby('Orientation').mean()
    charROI['PrefOri'] = tmpAvg.idxmax(axis=0)
    
    print(str(len(np.where(charROI['VisuallyEvoked']==True)[0])) + ' out of '
          + str(nROI_init) + ' ROIs are visually responsive ('
          + str(np.round( 100*len(
              np.where(charROI['VisuallyEvoked']==True)[0]
              )/nROI_init).astype(int) )+'%)')
    
    # Center of mass of ROIs
    for n in range(nROI_init):
        this_xCM, this_yCM = ndimage.center_of_mass(positionROI_3d[n])
        this_xCM = int(np.round(this_xCM))
        this_yCM = int(np.round(this_yCM))
        if n==0:
            cmROI = np.array([[this_xCM, this_yCM]])
        else:
            cmROI = np.append(cmROI, np.array([[this_xCM, this_yCM]]), axis=0)
            
    charROI['xCM'] = pd.DataFrame(cmROI[:,0])
    charROI['yCM'] = pd.DataFrame(cmROI[:,1])
    
    # Compute pairwise distance between all ROIs
    distROI = pixelSize*pairwise.euclidean_distances(cmROI)
    
    # Find pairs of ROIs which are too close to each other
    tmp_idxROItooClose = np.argwhere(distROI<globalParams.threshold_distance)
    
    # Size of ROIs (unit: pixels)
    charROI['Size'] = positionROI.sum(axis=1)
    
    # Discard smallest ROI of a pair which is too close
    idxROItooClose = []
    for i in range(tmp_idxROItooClose.shape[0]):
        this_pair = tmp_idxROItooClose[i,:]
        if this_pair[0] != this_pair[1]:
            if (~np.isin(this_pair[0], idxROItooClose) 
                & ~np.isin(this_pair[1], idxROItooClose)):
                tmpSize0 = charROI['Size'][this_pair[0]]
                tmpSize1 = charROI['Size'][this_pair[1]]
                if tmpSize0 < tmpSize1:
                    idxROItooClose = np.append(idxROItooClose, this_pair[0])
                else:
                    idxROItooClose = np.append(idxROItooClose, this_pair[1])
    
    print('There are ' + str(len(idxROItooClose)) + ' ROIs which are discarded'
          + ' because they are too close from another (bigger) ROI')
    charROI['farEnough'] = np.array(
        [False if i in idxROItooClose else True for i in range(nROI_init)]
        )
    
    # Anova test to determine orientation selectivity
    group0 = avgFluo_Stimulus.loc[globalParams.ori[0]]
    group1 = avgFluo_Stimulus.loc[globalParams.ori[1]]
    group2 = avgFluo_Stimulus.loc[globalParams.ori[2]]
    group3 = avgFluo_Stimulus.loc[globalParams.ori[3]]
    f_val, p_val = stats.f_oneway(group0, group1, group2, group3)
    bool_OS = (p_val < globalParams.threshold_pval_OS)
    
    # Not visually evoked ROIs cannot be orientation selective
    idxNotVE = np.where( (charROI['VisuallyEvoked']==False) & (bool_OS==True)
                        )[0]
    bool_OS[idxNotVE] = 0
    
    # Determine orientation-selectivity of ROIs
    nOS = len( np.where(bool_OS==True)[0] )
    nVE = len( np.where(charROI['VisuallyEvoked']==True)[0] )
    print('There are ' + str(nOS) + ' visually evoked ROIs which are '
          + 'orientation selective (' 
          + str( np.round(100*nOS/nVE ).astype(int)) + '%)')
    charROI['OS'] = np.array(
        [True if bool_OS[i]==1 else False for i in range(nROI_init)]
        )
    
    # Determine which ROIs we keep for the analyses
    charROI['keptROI'] = charROI['farEnough']
    
    # Plot all ROIs
    plt.figure()
    plt.imshow( np.transpose( np.sum(positionROI_3d, axis=0) ) )
    plt.title('All ROIs')
    
    idxNotVR = np.where( charROI['VisuallyEvoked']==False )[0]
    plt.figure()
    plt.imshow( np.transpose( np.sum(positionROI_3d[idxNotVR], axis=0) ) )
    plt.title('Not visually-evoked ROIs')
    
    # Discard ROIs
    idxKept = np.where( charROI['keptROI']==True )[0]
    fluo_array = fluo_array[:,idxKept]
    fluo = pd.DataFrame(fluo_array)
    positionROI = pd.DataFrame( np.array(positionROI)[idxKept,:] )
    positionROI_3d = positionROI_3d[idxKept,:,:]
    distROI = distROI[idxKept,:][:,idxKept]
    charROI = charROI[charROI['keptROI']==True].reset_index()
    charROI = charROI.drop(columns=['index','farEnough','keptROI'])
    data = pd.concat([charTrials, fluo], axis=1)
    
    # Number of ROIs we keep after postprocessing
    nROI = charROI.shape[0]

    # Plot all kept ROIs at the end of postprocessing
    plt.figure()
    plt.imshow( np.transpose( np.sum(positionROI_3d, axis=0) ) )
    plt.title('ROIs kept after postprocessing')
    
    # Plot all kept ROIs with their selectivity
    plot_ROIwithSelectivity(charROI, positionROI, fluoPlaneWidth, 
                            fluoPlaneHeight)
    
    # Plot the average fluorescence over all ROIs for each trial orientation
    plot_avgFluoPerOri(data, dataFramesPerTrial)
    
    # Plot the average fluorescence over only the OS ROIs for the corresponding
    # trial orientation
    idxOS = np.where( charROI['OS']==True )[0]
    fluo_array = fluo_array[:,idxOS]
    fluoOS = pd.DataFrame(fluo_array)
    dataOS = pd.concat([charTrials, fluoOS], axis=1)
    plot_avgFluoPerOri(dataOS, dataFramesPerTrial)
    
    print('We keep ' + str(nROI) + ' ROIs out of ' + str(nROI_init)
          +' recorded ROIs (' 
          + str( np.round(100*nROI/nROI_init).astype(int) ) + '%).')
        
    return charROI, charTrials, fluo, data, positionROI, distROI, idxKept


def plot_ROIwithSelectivity(charROI, positionROI, fluoPlaneWidth, 
                            fluoPlaneHeight):
    """Plot the ROIs color-coded according to their orientation selectivity."""
    
    positionROI = np.array(positionROI)
    positionROI_3d = positionROI.reshape((positionROI.shape[0], fluoPlaneWidth,
                                          fluoPlaneHeight))
    
    coloredROI = np.zeros( (fluoPlaneHeight, fluoPlaneWidth) )
    for i in range( len(globalParams.ori) ):
        tmpIdx = np.where((charROI['OS']==True) 
                          & (charROI['PrefOri']==globalParams.ori[i]))[0]
    
        if tmpIdx.size==1:
            tmp = np.clip((i+1)*positionROI_3d[tmpIdx], 0, i+1 )
        else:
            tmp = np.clip(np.sum((i+1)*positionROI_3d[tmpIdx], axis=0), 0, i+1)
        coloredROI = np.clip(coloredROI + tmp, 0, i+1)
    
    # Visually responsive non-OS ROIs
    tmpIdx = np.where((charROI['VisuallyEvoked']==True) 
                      & (charROI['OS']==False)
                      )[0]
    if tmpIdx.size==1:
        tmp = np.clip((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:], 
                      0, len(globalParams.ori)+1)
    else:
        tmp = np.clip(np.sum( 
            (len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:], axis=0
            ), 0, len(globalParams.ori)+1)
    coloredROI = np.clip(coloredROI + tmp, 0, len(globalParams.ori)+1)
    
    # Clip the values for overlaps
    coloredROI = np.clip(coloredROI, 0, len(globalParams.ori)+1)
    
    # Non-visually responsive ROIs (=spontaneously active ROIs)
    tmpIdx = np.where(charROI['VisuallyEvoked']==False)[0]
    if tmpIdx.size==1:
        tmp = np.clip((globalParams.nOri+2)*positionROI_3d[tmpIdx,:,:],
                      0, len(globalParams.ori)+2)
    else:
        tmp = np.clip(np.sum((globalParams.nOri+2)*positionROI_3d[tmpIdx,:,:],
                             axis=0),
                      0, len(globalParams.ori)+2)
    coloredROI = np.clip(coloredROI + tmp, 0, len(globalParams.ori)+2)
    
    # Create colormap
    cmap = colors.ListedColormap(['white', 'red', 'green', 'blue', 'orange',
                                  'black', 'grey'])
    bounds = np.linspace(0, 7, 8)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    plt.figure()
    # tell imshow about color map so that only set colors are used
    img = plt.imshow(np.transpose(coloredROI), interpolation='nearest', 
                     cmap=cmap, norm=norm)
    
    # make a color bar (it uses 'cmap' and 'norm' form above)
    cbar = plt.colorbar(img, boundaries=np.linspace(1,7,7),
                        ticks=np.linspace(1.5, 6.5, 6))
    cbar.ax.set_yticklabels(['45°', '135°','180°','270°','non-OS','spont'])

    plt.title('Kept ROIs with orientation selectivity')
    #plt.savefig('ROIwithSel.eps', format='eps')


# Plot all kept V1 ROIs and thalamic axons at the end of preprocessing
def plot_ROIwithBoutons(positionROI, positionBoutons, fluoPlaneWidth, 
                        fluoPlaneHeight, idxOverlappingBoutons=None):
    
    positionROI = np.array(positionROI)
    positionBoutons = np.array(positionBoutons)
    positionROI_3d = positionROI.reshape((positionROI.shape[0], fluoPlaneWidth,
                                          fluoPlaneHeight))
    positionBoutons_3d = positionBoutons.reshape((positionBoutons.shape[0],
                                                  fluoPlaneWidth, fluoPlaneHeight))
    
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


def plot_avgFluoPerOri(data, dataFramesPerTrial):
    """Plot the average fluorescence for each trial orientation."""
    
    tmpTime = np.arange(dataFramesPerTrial)+1
    tmp = data.drop(['Session','Trial','FrameType'],axis=1)
    avgPerOri = tmp.groupby(['TrialFrame','Orientation']).mean()
    avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
    avgPerOri = avgPerOri.reset_index()
    
    fig, axs = plt.subplots(2, 2)
    o = 0
    for x in range(2):
        for y in range(2):
            tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[o]]
            tmp = np.array(tmp.drop(['TrialFrame','Orientation'], axis=1))
            tmpMean = np.mean(tmp, axis=1)
            tmpSEM = np.std(tmp, axis=1)/np.sqrt(tmp.shape[1])
            axs[x,y].plot(tmpTime, tmpMean)
            axs[x,y].fill_between(tmpTime, tmpMean-tmpSEM, tmpMean+tmpSEM)
            axs[x,y].set_title('Orientation: ' + str(globalParams.ori[o]) + '°')
            o +=1


def plot_avgPupilPerOri(data, dataFramesPerTrial):
    """Plot the average pupil area for each trial orientation."""
    
    tmpTime = np.arange(dataFramesPerTrial)+1
    tmp = data.drop(['Session','Trial','FrameType'],axis=1)
    avgPerOri = tmp.groupby(['TrialFrame','Orientation']).mean()
    avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
    avgPerOri = avgPerOri.reset_index()
    semPerOri = tmp.groupby(['TrialFrame','Orientation']).sem()
    semPerOri = semPerOri.sort_values(['Orientation','TrialFrame'])
    semPerOri = semPerOri.reset_index()
    
    fig, axs = plt.subplots(2, 2)
    o = 0
    for x in range(2):
        for y in range(2):
            tmp = avgPerOri[avgPerOri['Orientation']==globalParams.ori[o]]
            tmpD = tmp.drop(['TrialFrame','Orientation'],axis=1)
            tmpMean = tmpD.values.flatten()
            tmp2 = semPerOri[semPerOri['Orientation']==globalParams.ori[o]]
            tmp2D = tmp2.drop(['TrialFrame','Orientation'],axis=1)
            tmpSEM = tmp2D.values.flatten()
            axs[x,y].plot(tmpTime,tmpMean)
            axs[x,y].fill_between(tmpTime, tmpMean-tmpSEM, tmpMean+tmpSEM)
            axs[x,y].set_title('Orientation: ' + str(globalParams.ori[o])
                               + '°')
            o +=1
    fig.suptitle('Pupil area')


def postprocess_pupil(dataType, dataDate, dataMouse, dataDepth):
    """ Postprocess pupil data."""

    pupilfilepath = (globalParams.dataDir + dataType +'\\' + dataDate + '_'
                     + dataMouse + '\\' + dataDepth + '\\pupil_manual.mat')
    
    if not os.path.isfile(pupilfilepath):
        pupilfilepath = (globalParams.dataDir + dataType +'\\' + dataDate + '_' 
                         + dataMouse + '\\' + dataDepth + '\\pupil.mat')
    
    if os.path.isfile(pupilfilepath):
        
        print("Taking care of pupil data...")
        
        # Read pupil data
        if (dataType=='L4_cytosolic') | (dataType=='L23_thalamicBoutons'):
            
            f = scipy.io.loadmat(pupilfilepath)
            tmp = f['pupil']
            val = tmp[0,0]
            # majorAxisLength = val['MajorAxisLength']
            # minorAxisLength = val['MinorAxisLength']
            pupilArea = val['pupilArea']
            # xCenter = val['Xc']
            # yCenter = val['Yc']
            
        elif dataType=='L4_LGN_targeted_axons':
            
            arrays = {}
            f = h5py.File(pupilfilepath,'r')
            for k, v in f.items():
                arrays[k] = np.array(v)    
            pupilArea = arrays['pupilArea'][0]
            
            # Discard the last 30 frames of each session (they occured after 
            # the end of the 200 trials)
            pupilArea = np.reshape(pupilArea, (2430,-1), order='F')
            pupilArea = np.delete(pupilArea, np.arange(2400,2430), axis=0)
            pupilArea = pupilArea.flatten(order='F')
            
        charPupil = pd.DataFrame(np.transpose(pupilArea), 
                                 columns=['pupilArea'])
        # charPupil['majorAxisLength'] = np.transpose(majorAxisLength)
        # charPupil['minorAxisLength'] = np.transpose(minorAxisLength)
        # charPupil['xCenter'] = np.transpose(xCenter)
        # charPupil['yCenter'] = np.transpose(yCenter)
        
    else:
        charPupil = None
        
    return charPupil


def postprocess_motion(dataType,dataDate,dataMouse,dataDepth):
    """Postprocess motion data."""

    motionfilepath = (globalParams.dataDir + dataType +'\\' + dataDate + '_' 
                      + dataMouse + '\\' + dataDepth + '\\motion.mat')
    
    if os.path.isfile(motionfilepath):
        
        print("Taking care of motion data...")
        
        f = scipy.io.loadmat(motionfilepath)
        tmp = f['motion']
        val = tmp[0,0]
        motSVD = pd.DataFrame(val['motSVD'])
        motAvg = val['mot_avg']
        
        if dataType=='L4_LGN_targeted_axons':
            # Discard the last 30 frames of each session (they occured after
            # the end of the 200 trials)
            z = []
            for i in range(5):
                z.append(i*2400+np.arange(2400,2430,1))
            z = np.array(z).flatten()
            motSVD = motSVD.drop(index=z).reset_index(drop=True)

        # Plot the average motion
        plt.figure()
        plt.imshow(motAvg)
        
    else:
        motSVD = None
            
    return motSVD


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
    
    # Center of mass of ROIs
    for n in range(dff0.shape[0]):
        this_xCM, this_yCM = ndimage.center_of_mass(positionROI_3d[n])
        this_xCM = int(np.round(this_xCM))
        this_yCM = int(np.round(this_yCM))
        if n==0:
            cmROI = np.array([[this_xCM,this_yCM]])
        else:
            cmROI = np.append(cmROI,np.array([[this_xCM,this_yCM]]),axis=0)
            
    charROI['xCM'] = pd.DataFrame(cmROI[:,0])
    charROI['yCM'] = pd.DataFrame(cmROI[:,1])
    
    # Compute pairwise distance between all ROIs
    distROI = pixelSize*pairwise.euclidean_distances(cmROI)
    
    ### start 2
    # Anova test to determine orientation selectivity
    group0 = avgFluo_Stimulus.loc[globalParams.ori[0]]
    group1 = avgFluo_Stimulus.loc[globalParams.ori[1]]
    group2 = avgFluo_Stimulus.loc[globalParams.ori[2]]
    group3 = avgFluo_Stimulus.loc[globalParams.ori[3]]
    f_val, p_val = stats.f_oneway(group0, group1, group2, group3)
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







