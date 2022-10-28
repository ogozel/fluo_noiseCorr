# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:11:45 2022

@author: Olivia Gozel

Investigate the spectral power structure of the face motion and pupil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal


os.chdir('C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\Code_python\\')

import globalParams
import functions_analyze




#%% Parameters of the postprocessed data to analyze

### TO CHOOSE ###
dataType = 'L23_thalamicBoutons' # 'L4_LGN_targeted_axons' or 'L4_cytosolic' or 'L23_thalamicBoutons'
boolV1 = True
dataNeuropilSub = globalParams.neuropilSub[3] # choose a neuropil factor
# globalParams.neuropilSub[3] for V1 data
# globalParams.neuropilSub[0] for thalamic boutons

filepath = globalParams.dataDir + dataType + '_dataSpecs.hdf'
dataSpecs = pd.read_hdf(filepath,dataType+'_dataSpecs')



#%% Load data

idxDataset = 2

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

allData = pd.HDFStore(filepath)
fluo = pd.read_hdf(allData,'fluo')
charTrials = pd.read_hdf(allData,'charTrials')
charROI = pd.read_hdf(allData,'charROI')
if 'charPupil' in allData:
    charPupil = pd.read_hdf(allData,'charPupil')
if 'motSVD' in allData:
    motSVD = pd.read_hdf(allData,'motSVD')


#%% Spectral analysis of pupil area - for L4 thalamic boutons

frameRate = 2 # Hz for pupil and face motion
NyquistFreq = frameRate/2

# Each recording session has to be analyzed separately since they did not occur directly one after the other
sig = np.squeeze(np.array(charPupil))[0:2400]
#sig = np.squeeze(np.array(motSVD[0]))[0:2400]

freqs, psd = signal.welch(sig,fs=frameRate,scaling='spectrum')

plt.figure()
#plt.semilogx(freqs, psd)
plt.semilogy(freqs, psd)
#plt.plot(freqs, psd)
plt.title('Power spectral density - pupilArea')
plt.xlabel('Frequency')
plt.ylabel('Power')
#plt.tight_layout()
plt.show()


#%% Detrend fluorescence data - for L2/3 ROIs, dataset2 for example

plt.figure()
plt.plot(np.mean(np.array(fluo),axis=1))
plt.xlabel('Frame')
plt.ylabel('Avg dff0')
plt.title('Before detrending')

# Only look at one session
thisFluo = fluo.loc[charTrials['Session']==1]

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

plt.figure()
plt.plot(np.mean(np.array(thisFluo),axis=1))
plt.xlabel('Frame')
plt.ylabel('Avg dff0')
plt.title('Before detrending')

Master_cutoff = [1e-30, 1e-20, 1e-15, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for i in range(len(Master_cutoff)):
    filtered_thisFluo = butter_highpass_filter(thisFluo, Master_cutoff[i], dataFR)
    
    plt.figure()
    plt.plot(np.mean(filtered_thisFluo,axis=1))
    plt.xlabel('Frame')
    plt.ylabel('Avg dff0')
    plt.title('After detrending <'+str(Master_cutoff[i])+' Hz')


