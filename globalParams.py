# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:44:34 2021

@author: Olivia Gozel

Define global variables
"""

# Path to data
projectDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\'
dataDir = projectDir + 'U19data\\'
processedDataDir = projectDir + 'U19data_postprocessed\\'

# Grating orientations in degrees
ori = (45, 135, 180, 270)
nOri = len(ori)
# Name extension for neuropil subtraction
neuropilSub = ('NpMethod0',\
               'NpMethod1_Coe0.25_Exclusion_NpSize30',\
               'NpMethod1_Coe0.5_Exclusion_NpSize30',\
               'NpMethod1_Coe0.75_Exclusion_NpSize30',\
               'NpMethod1_Coe1_Exclusion_NpSize30')
# Minimal distance between pairs of ROIs in [um]
threshold_distance = 5
# p-value below which a neuron is considered orientation selective
threshold_pval_OS = 0.05 
# p-value below which a neuron is considered visually evoked
threshold_pval_visuallyEvoked = 0.05
