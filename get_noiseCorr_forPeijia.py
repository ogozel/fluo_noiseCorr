# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:05:12 2022

@author: Olivia Gozel

Get the noise correlations for 5 ROIs for Peijia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze
import functions_preprocess




#%% Parameters of the preprocessed data to analyze

### TO CHOOSE ###
dataType = 'L23_thalamicBoutons' # 'L4_cytosolic' or 'L23_thalamicBoutons'

filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')

### TO CHOOSE ###
idxDataset = 1 # L4_cytosolic: 0,1,4,8: nice corr=f(dist); 2,3,5,6,7,9: bad corr=f(dist)

dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
dataFR = dataSpecs.iloc[idxDataset]['FrameRate']

if dataType=='L23_thalamicBoutons':
    dataFramesPerTrial = dataSpecs.iloc[idxDataset]['nFramePerTrial']
elif dataType=='L4_cytosolic':
    dataFramesPerTrial = globalParams.nFramesPerTrial

### To CHOOSE ###
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor


#%% Load data

# filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
#         dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist2d5um.hdf'
filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist10um.hdf'

fluo = pd.read_hdf(filepath,'fluo')
charROI = pd.read_hdf(filepath,'charROI')
charTrials = pd.read_hdf(filepath,'charTrials')
positionROI = pd.read_hdf(filepath,'positionROI')
distROI = pd.read_hdf(filepath,'distROI')
fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()


#%% All ROIs

functions_preprocess.plot_ROIwithSelectivity(charROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)


#%% Only OS ROIs

fluo = fluo[np.where(charROI['OS']==True)[0]]
distROI = distROI[np.where(charROI['OS']==True)[0]].loc[np.where(charROI['OS']==True)[0]]

fluo_meanSub = np.array(fluo - np.mean(fluo,axis=0))
zcorr = np.corrcoef(fluo_meanSub, rowvar=False)


#%% Find some good ROIs

tmpIdx = np.where((charROI['OS']==True) & (charROI['PrefOri']==45))[0]

# Pairwise distances between ROIs we select
zdist = distROI[tmpIdx].loc[tmpIdx]

# Pairwise correlations
zFluo = fluo[tmpIdx]
fluo_meanSub = np.array(zFluo - np.mean(zFluo,axis=0))
zcorr = np.corrcoef(fluo_meanSub, rowvar=False)

tmpIdx = np.where((charROI['OS']==True) & (charROI['PrefOri']==135))[0]
tmpIdx = np.concatenate((np.array([5]),tmpIdx))

# Pairwise distances between ROIs we select
zdist = distROI[tmpIdx].loc[tmpIdx]

# Pairwise correlations
zFluo = fluo[tmpIdx]
fluo_meanSub = np.array(zFluo - np.mean(zFluo,axis=0))
zcorr = np.corrcoef(fluo_meanSub, rowvar=False)


tmpIdx = [5, 40, 20, 55, 88]



#%% 

tmpIdx = np.where((charROI['xCM']>1200) & (charROI['xCM']<1700) & (charROI['yCM']>250) & (charROI['yCM']<1250))[0]
functions_preprocess.plot_ROIwithSelectivity(charROI.loc[tmpIdx],positionROI.loc[tmpIdx],fluoPlaneWidth,fluoPlaneHeight)


#%% ROIs which we select

tmpIdx = [93, 92, 104, 120, 96]

# Pairwise distances between ROIs we select
zdist = distROI[tmpIdx].loc[tmpIdx]

# Pairwise correlations
zfluo = fluo[tmpIdx]

# Mean-subtract the neural activity
fluo_meanSub = np.array(zfluo - np.mean(zfluo,axis=0))

# Full neural covariance matrix
corr_full = np.corrcoef(fluo_meanSub, rowvar=False)


fluo270 = zfluo.loc[charTrials['Orientation']==270]
fluo270 = np.array(fluo270)

for r in range(5):
    plt.plot(fluo270[0:30,r])


#%%


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
    #plt.figure()
    #plt.imshow(np.transpose(np.sum((i+1)*positionROI_3d[tmpIdx],axis=0)))

# Non-OS kept ROIs
tmpIdx = np.squeeze(np.array(np.where((charROI['OS']==False))))
if tmpIdx.size==1:
    tmp = np.clip((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],0,len(globalParams.ori)+1)
else:
    tmp = np.clip(np.sum((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],axis=0),0,len(globalParams.ori)+1)
coloredROI = np.clip(coloredROI + tmp, 0, len(globalParams.ori)+1)
#plt.figure()
#plt.imshow(np.transpose(np.sum((len(globalParams.ori)+1)*positionROI_3d[tmpIdx,:,:],axis=0)))

# Clip the values for overlaps
coloredROI = np.clip(coloredROI,0,len(globalParams.ori)+1)

# Create colormap
cmap = colors.ListedColormap(['white', 'red', 'green', 'blue', 'orange', 'black'])
bounds = np.linspace(0,6,7)
norm = colors.BoundaryNorm(bounds, cmap.N)

plt.figure()
# tell imshow about color map so that only set colors are used
img = plt.imshow(np.transpose(coloredROI), interpolation='nearest', cmap=cmap, norm=norm)

# make a color bar (it uses 'cmap' and 'norm' form above)
cbar = plt.colorbar(img, boundaries=np.linspace(1,6,6), ticks=np.linspace(1.5,5.5,5))
cbar.ax.set_yticklabels(['45°', '135°','180°','270°','non-OS'])

