# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:44:49 2021

@author: Olivia Gozel

Create a dataframe with all the data specifications
"""

import pandas as pd


dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data\\'


#%% L4_cytosolic data

L4_cytosolic_dataSpecs = pd.DataFrame()

L4_cytosolic_dataSpecs['Date'] = ['20200729','20200729','20201015','20201015','20201016',\
                                  '20201018','20201018','20201021','20201021',\
                                      '20201022','20201022']
L4_cytosolic_dataSpecs['Mouse'] = ['Y24','Y24','Y35','Y35','Y35',\
                                   'Y35','Y35','Y36','Y36',\
                                       'Y36','Y36']
L4_cytosolic_dataSpecs['Depth'] = ['Z280','Z310','Z330','Z360','Z290',\
                                   'Z330','Z360','Z300','Z330',\
                                       'Z360','Z390']
L4_cytosolic_dataSpecs['Sessions'] = [[2,3,4,5,6],[1,2,3,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],\
                                      [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],\
                                          [2,3,4,5,6],[1,2,3,4,5]]
L4_cytosolic_dataSpecs['PixelSize'] = [0.813,0.813,1,1,1,\
                                       1,1,1,1,\
                                           1,1]
L4_cytosolic_dataSpecs['FrameRate'] = [15,15,7.5,7.5,7.5,\
                                       7.5,7.5,7.5,7.5,\
                                           7.5,7.5]


savefilepath = dataDir + 'L4_cytosolic_dataSpecs.hdf'

L4_cytosolic_dataSpecs.to_hdf(savefilepath,key='L4_cytosolic_dataSpecs')




#%% L2/3 cytosolic data + thalamic bouton inputs

# NB: Dataset '20210106_Y45' does not have the thalamic boutons

L23_thalamicBoutons_dataSpecs = pd.DataFrame()

L23_thalamicBoutons_dataSpecs['Date'] = ['20210105','20210106','20210116','20210116']

L23_thalamicBoutons_dataSpecs['Mouse'] = ['Y45','Y45','Y43','Y43']

L23_thalamicBoutons_dataSpecs['Depth'] = ['Z150','Z120','Z100','Z120']

L23_thalamicBoutons_dataSpecs['Sessions'] = [[2,3,4,5,6],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]

L23_thalamicBoutons_dataSpecs['PixelSize'] = [0.3, 0.3, 0.3, 0.3]

L23_thalamicBoutons_dataSpecs['FrameRate'] = [7.6, 7.6, 7.6, 7.6]


savefilepath = dataDir + 'L23_thalamicBoutons_dataSpecs.hdf'

L23_thalamicBoutons_dataSpecs.to_hdf(savefilepath,key='L23_thalamicBoutons_dataSpecs')


