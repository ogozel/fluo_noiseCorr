# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:44:34 2021

@author: Olivi Gozel

Define global variables
"""

# Path to data
dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data\\'
processedDataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data_postprocessed\\'

# Fluorescence data
nFramesPerTrial = 25
nBlankFrames = 5
nStimFrames = 20
ori = (45, 135, 180, 270) # grating orientations in degrees
nOri = 4
pixelSize = 1 # [um] default value
neuropilSub = ('NpMethod0',\
               'NpMethod1_Coe0.25_Exclusion_NpSize30',\
               'NpMethod1_Coe0.5_Exclusion_NpSize30',\
               'NpMethod1_Coe0.75_Exclusion_NpSize30',\
               'NpMethod1_Coe1_Exclusion_NpSize30')

threshold_dff0 = 10 # [%]
threshold_distance = 2.5 # [um] (to merge ROIs that are too close to each others)
threshold_pval = 0.05 # pvalue below which a neuron is considered orientation selective
threshold_percentile = 0.15 # percentile for baselineRaw under which ROIs are fake