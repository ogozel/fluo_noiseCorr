# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:46:09 2021

@author: Olivia Gozel

Get the pairwise correlation as a function of distance data for Peijia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze

dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data\\'

peijiaDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Peijia\\'


#%% Parameters of the preprocessed data to analyze

dataType = 'L4_cytosolic' # 'L4_cytosolic' or 'L23_thalamicBoutons'
idxDataset = 0
# L4 good: 0,1,2,3,4,5,6,7,8,9
# L2/3: 2 is weird (very high pairwise correlation at long distances)

filepath = dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath)


dataDate = dataSpecs.iloc[idxDataset]['Date']
dataMouse = dataSpecs.iloc[idxDataset]['Mouse']
dataDepth = dataSpecs.iloc[idxDataset]['Depth']
pixelSize = dataSpecs.iloc[idxDataset]['PixelSize']
dataSessions = dataSpecs.iloc[idxDataset]['Sessions']
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor, by default: globalParams.neuropilSub[3]
# NB: dataType='L23_thalamicBoutons' with idxDataset=2 also has globalParams.neuropilSub[4]


#%% Load data

filepath = globalParams.processedDataDir + dataType +'_' + dataDate + '_' + \
        dataMouse + '_' + dataDepth + '_neuropilF_' + dataNeuropilSub + '_threshDist10um.hdf'

fluo = pd.read_hdf(filepath,'fluo')
charROI = pd.read_hdf(filepath,'charROI')
charTrials = pd.read_hdf(filepath,'charTrials')
positionROI = pd.read_hdf(filepath,'positionROI')
distROI = pd.read_hdf(filepath,'distROI')



#%% Noise pairwise correlation as a function of pairwise distance
# Neuropil factor = 0.75
# 5um distance bins, first one is between 5um and 10um (centered in 7.5um)
# OS ROIs only
# Stimulus frames only

binCenters, sortedPairCorr = functions_analyze.plot_noiseCorr_fdist(fluo,distROI,charROI,charTrials,boolVisuallyEvoked=True,boolOS=True,boolIdenticalTuning=False,boolStimulus=True,startBin=5,binSize=5)

# Save it in excel
df = pd.DataFrame(sortedPairCorr)
#writer = pd.ExcelWriter(peijiaDir+'L23cytosolic_pairCorr_fDist.xlsx',engine='openpyxl') # when the file does not exist yet
writer = pd.ExcelWriter(peijiaDir+'L23cytosolic_pairCorr_fDist.xlsx',engine='openpyxl',mode='a') # once the file exists
df.to_excel(writer, sheet_name='Dataset'+str(idxDataset))
writer.save()



#%% Noise pairwise correlation as a function of pairwise distance for similarly tuned ROIs
# Neuropil factor = 0.75
# 5um distance bins, first one is between 5um and 10um (centered in 7.5um)
# OS ROIs only, separately for each orientation selectivity
# Stimulus frames only
# For pairs of OS ROIs with same orientation tuning (DeltaTheta=0), combined for all four orientations

binCenters, sortedPairCorr = functions_analyze.plot_noiseCorr_fdist(fluo,distROI,charROI,charTrials,boolVisuallyEvoked=True,boolOS=True,boolIdenticalTuning=True,boolStimulus=True,startBin=5,binSize=5)


# Save it in excel
df = pd.DataFrame(sortedPairCorr)
#writer = pd.ExcelWriter(peijiaDir+'L23cytosolic_pairCorr_fDist_sameTuning.xlsx',engine='openpyxl') # when the file does not exist yet
writer = pd.ExcelWriter(peijiaDir+'L23cytosolic_pairCorr_fDist_sameTuning.xlsx',engine='openpyxl',mode='a') # once the file exists
df.to_excel(writer, sheet_name='Dataset'+str(idxDataset))
writer.save()


#%% Select 5 ROIs

nFramesPerTrial = 30
nTrialsPerOri = 250
idxT135 = np.where(charTrials['Orientation']==135)[0]
ref_ROI = 87 # prefOri=135°
close_sameTuning_ROI = 90 # prefOri=135°, 8.7um apart
close_diffTuning_ROI = 14 # prefOri=270°, 19.7um apart
far_sameTuning_ROI = 82 #142 # prefOri=135°, 83.8um apart # 72.4um apart
far_diffTuning_ROI = 100 # prefOri=270°, 84.0um apart # 63.1um apart

# Trials of the preferred orientation of the reference ROI
thisFluo_ref = np.array(fluo[ref_ROI].loc[idxT135])
thisFluo_closeST = np.array(fluo[close_sameTuning_ROI].loc[idxT135])
thisFluo_closeDT = np.array(fluo[close_diffTuning_ROI].loc[idxT135])
thisFluo_farST = np.array(fluo[far_sameTuning_ROI].loc[idxT135])
thisFluo_farDT = np.array(fluo[far_diffTuning_ROI].loc[idxT135])

print('Correlation of ref with close sameTuning: '+str(np.corrcoef(thisFluo_ref - np.mean(thisFluo_ref),thisFluo_closeST- np.mean(thisFluo_closeST))[0,1]))
print('Correlation of ref with close diffTuning: '+str(np.corrcoef(thisFluo_ref - np.mean(thisFluo_ref),thisFluo_closeDT- np.mean(thisFluo_closeDT))[0,1]))
print('Correlation of ref with far sameTuning: '+str(np.corrcoef(thisFluo_ref - np.mean(thisFluo_ref),thisFluo_farST- np.mean(thisFluo_farST))[0,1]))
print('Correlation of ref with far diffTuning: '+str(np.corrcoef(thisFluo_ref - np.mean(thisFluo_ref),thisFluo_farDT- np.mean(thisFluo_farDT))[0,1]))


theseTrials = np.arange(0,100,1) #np.array((3,4,12,13,20))
for i in range(len(theseTrials)):
    thisT = theseTrials[i]
    plt.figure()
    plt.plot(thisFluo_ref[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    plt.plot(thisFluo_closeST[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    plt.plot(thisFluo_closeDT[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    plt.plot(thisFluo_farST[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    plt.plot(thisFluo_farDT[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    plt.legend(['ref','close sameTuning', 'close diffTuning','far sameTuning', 'far diffTuning'])
    plt.title('Trial='+str(thisT));
    

# Plot individual fluorescence traces for the 5 ROIs of interest, for a selection of trials
theseTrials = np.array((3,22,30,39,62,63,82,85))
for i in range(len(theseTrials)):
    thisT = theseTrials[i]
    fig,axs = plt.subplots(1,5, constrained_layout=True, sharey=True)
    axs[0].plot(thisFluo_ref[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    axs[0].set_title(str(ref_ROI))
    axs[1].plot(thisFluo_closeST[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    axs[1].set_title(str(close_sameTuning_ROI))
    axs[2].plot(thisFluo_closeDT[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    axs[2].set_title(str(close_diffTuning_ROI))
    axs[3].plot(thisFluo_farST[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    axs[3].set_title(str(far_sameTuning_ROI))
    axs[4].plot(thisFluo_farDT[thisT*nFramesPerTrial:(thisT+1)*nFramesPerTrial]);
    axs[4].set_title(str(far_diffTuning_ROI))
    #fig.suptitle('Trial='+str(thisT));
    fig.savefig(peijiaDir+'trial'+str(thisT)+'.eps', format='eps')


# Plot the average fluorescence trace for each orientation for the ROIs of interest
thisROI = 100
data = pd.concat([charTrials[['TrialFrame','Orientation']],fluo[thisROI]],axis=1)
avgPerOri = data.groupby(['TrialFrame','Orientation']).mean()
avgPerOri = avgPerOri.sort_values(['Orientation','TrialFrame'])
avgPerOri = avgPerOri.reset_index()
stdPerOri = data.groupby(['TrialFrame','Orientation']).std()
stdPerOri = stdPerOri.sort_values(['Orientation','TrialFrame'])
stdPerOri = stdPerOri.reset_index()

fig, axs = plt.subplots(2, 2, constrained_layout=True, sharey=True)
axs = axs.ravel()
for o in range(globalParams.nOri):
    tmp_avg = np.squeeze(np.array(avgPerOri[avgPerOri['Orientation']==globalParams.ori[o]].drop(['TrialFrame','Orientation'],axis=1)))
    tmp_std = np.squeeze(np.array(stdPerOri[stdPerOri['Orientation']==globalParams.ori[o]].drop(['TrialFrame','Orientation'],axis=1)))
    tmp_sem = tmp_std/np.sqrt(nTrialsPerOri)
    axs[o].fill_between(np.arange(0,nFramesPerTrial),tmp_avg-tmp_sem,tmp_avg+tmp_sem)
    axs[o].set_title(str(globalParams.ori[o])+'°')
fig.suptitle('ROI '+str(thisROI))


# Plot the ROIs
import functions_postprocess
fluoPlaneWidthHeight = pd.read_hdf(filepath,'fluoPlaneWidthHeight')
fluoPlaneWidth = np.array(fluoPlaneWidthHeight)[0].item()
fluoPlaneHeight = np.array(fluoPlaneWidthHeight)[1].item()

# Hack to reuse function script (just need to change legend)
# cbar.ax.set_yticklabels(['90','14','82','100','87 (ref)','other'])
thisCharROI = charROI
thisCharROI['VisuallyEvoked'] = False
thisCharROI['VisuallyEvoked'].loc[ref_ROI] = True
thisCharROI['OS'].loc[ref_ROI] = False
thisCharROI['VisuallyEvoked'].loc[close_sameTuning_ROI] = True
thisCharROI['PrefOri'].loc[close_sameTuning_ROI] = 45
thisCharROI['VisuallyEvoked'].loc[close_diffTuning_ROI] = True
thisCharROI['PrefOri'].loc[close_diffTuning_ROI] = 135
thisCharROI['VisuallyEvoked'].loc[far_sameTuning_ROI] = True
thisCharROI['PrefOri'].loc[far_sameTuning_ROI] = 180
thisCharROI['VisuallyEvoked'].loc[far_diffTuning_ROI] = True
thisCharROI['PrefOri'].loc[far_diffTuning_ROI] = 270
functions_postprocess.plot_ROIwithSelectivity(thisCharROI,positionROI,fluoPlaneWidth,fluoPlaneHeight)


