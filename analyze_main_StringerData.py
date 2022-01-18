# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:12:25 2021

@author: Olivia Gozel

Analyze Stringer et al, Science (2019) data using the same home-made functions
"""
import scipy.io
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import os

os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze


dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\StringerData\\'


#%% Load data

filepath = dataDir + 'spont_M150824_MP019_20160405.mat'
# 'spont_M150824_MP019_20160405.mat'
# 'spont_M160907_MP028_20160926'
# 'spont_M160825_MP027_20161212'
# 'spont_M161025_MP030_20170616'


arrays = {}
f = scipy.io.loadmat(filepath)
for k, v in f.items():
    arrays[k] = np.array(v)

# Fluorescence data
fluo = np.transpose(arrays['Fsp']) # fluorescences values
positionROI = arrays['med'] # location of fluorescence data
allDepths = np.unique(positionROI[:,2]) # Depths of recordings [nm]

# Behavioral data
beh = arrays['beh']
val = beh[0,0]

# Pupil data
bool_pupil = True
pupil = val['pupil']
pupilArea = pupil['area'] #pupilArea = beh.pupil.area
pupilArea = np.squeeze(pupilArea[0,0])

# Motion data
bool_motion = True
motion = val['face']
motionSVD = motion['motionSVD'][0,0] #beh.face.motionSVD
#motionSVD = motionSVD[0,0]
motionSVD1 = motionSVD[:,0]

avgMotion= motion['avgframe'][0,0] #beh.face.avgframe
plt.figure()
plt.imshow(avgMotion)
plt.title('Average motion')





#%% Compute the neural PCs for each depth of recordings

neuralProjPerDepth = np.empty((fluo.shape[0],len(allDepths)))

for d in range(0,len(allDepths)):

    # Select the ROIs we are interested in
    idxThisDepth = np.squeeze(np.array(np.where(positionROI[:,2]==allDepths[d])))
    
    # Corresponding fluorescence values
    thisFluo = np.array(fluo)[:,idxThisDepth]

    # Compute neural projections for the current depth
    neuralProj,_ = functions_analyze.compute_neuralPCs(thisFluo)
    
    # Keep in memory only the projection on the first neural PC
    neuralProjPerDepth[:,d] = neuralProj[:,0]




#%% Define the traces of interest

traceNeuralperDepth = neuralProjPerDepth
tracePupil = pupilArea
traceMotion = motionSVD1

# Flip trace if average is negative
if np.mean(traceMotion) < 0:
    traceMotion = -traceMotion



#%% Plotting

# Plot the traces
functions_analyze.plot_traces(traceNeuralperDepth,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion,allDepths=allDepths)

# Plot auto-correlations
functions_analyze.plot_autocorrelations(traceNeuralperDepth,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion,allDepths=allDepths)

# Plot cross-correlations
functions_analyze.plot_crosscorrelations(traceNeuralperDepth,tracePupil,traceMotion,bool_pupil=bool_pupil,bool_motion=bool_motion,allDepths=allDepths)




#%% Plot the pairwise correlation as a function of pairwise distance

pixelSize = 1 # [nm]  no idea what the pixel size is in Stringer's data??

for d in range(0,len(allDepths)):

    # Select the ROIs we are interested in
    idxThisDepth = np.squeeze(np.array(np.where(positionROI[:,2]==allDepths[d])))
    
    # Corresponding fluorescence values
    thisFluo = np.array(fluo)[:,idxThisDepth]
    
    # Corresponding positions of CM
    thisPositionROI = positionROI[idxThisDepth,0:2]
    
    # Corresponding pairwise distances
    thisDistROI = pixelSize*euclidean_distances(thisPositionROI)
    
    # Plot correlation as a function of pairwise distance for eaach depth
    functions_analyze.plot_corr_fdist(thisFluo,thisDistROI,binSize=10,title='Stringers data - '+str(allDepths[d])+'nm')





