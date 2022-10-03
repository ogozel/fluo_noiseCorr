# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:44:34 2021

@author: Olivia Gozel

Define global variables
"""

# Path to data
dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data\\'
processedDataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data_postprocessed\\'

# Fluorescence data
ori = (45, 135, 180, 270) # grating orientations in degrees
nOri = 4
neuropilSub = ('NpMethod0',\
               'NpMethod1_Coe0.25_Exclusion_NpSize30',\
               'NpMethod1_Coe0.5_Exclusion_NpSize30',\
               'NpMethod1_Coe0.75_Exclusion_NpSize30',\
               'NpMethod1_Coe1_Exclusion_NpSize30')

# threshold_dff0 = 10 # [%]
threshold_distance = 5 # [um] (to merge ROIs that are too close to each others)
# NB: for Peijia: use threshold_distance = 2.5, then he asked 5
threshold_pval_OS = 0.05 # pvalue below which a neuron is considered orientation selective
threshold_pval_visuallyEvoked = 0.05
# threshold_percentile = 0.15 # percentile for baselineRaw under which ROIs are fake