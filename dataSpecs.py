# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:44:49 2021

@author: Olivia Gozel

Create a dataframe with all the data specifications
"""

import pandas as pd


dataDir = 'C:\\Users\\olivi\\Dropbox\\Projects\\U19_project\\U19data\\'


#%% L4_cytosolic data: datasets 0 to 10 included

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
L4_cytosolic_dataSpecs['FrameRate'] = [15.1, 15.1, 7.626, 7.626, 7.626,\
                                       7.626, 7.626,\
                                       7.626, 7.626, 7.626, 7.626]   

L4_cytosolic_dataSpecs['nBlankFrames'] = [5,5,5,5,7,\
                                          7, 7,\
                                          7, 7, 7, 7]

L4_cytosolic_dataSpecs['nStimFrames'] = [20,20,20,20,18,\
                                         18, 18,\
                                         18, 18, 18, 18]


savefilepath = dataDir + 'L4_cytosolic_dataSpecs.hdf'

L4_cytosolic_dataSpecs.to_hdf(savefilepath,key='L4_cytosolic_dataSpecs')




#%% L2/3 cytosolic data (+ thalamic boutons to L2/3 for some recording sessions): datasets 0 to 7 included

# NB: The following datasets have the thalamic boutons: [0,2,3,5,6]
# NB: boutons in dataset [5,6] seem to have a problem (avg fluo per orientation has missing values)

L23_thalamicBoutons_dataSpecs = pd.DataFrame()

L23_thalamicBoutons_dataSpecs['Date'] = ['20210105','20210106','20210116','20210116',\
                                         '20210120','20210123','20210123','20210228']

L23_thalamicBoutons_dataSpecs['Mouse'] = ['Y45','Y45','Y43','Y43',\
                                          'Y43','Y44','Y44','Y54']

L23_thalamicBoutons_dataSpecs['Depth'] = ['Z150','Z120','Z100','Z120',\
                                          'Z100','Z100','Z130','Z320']

L23_thalamicBoutons_dataSpecs['Sessions'] = [[2,3,4,5,6],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],\
                                             [1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]

L23_thalamicBoutons_dataSpecs['PixelSize'] = [0.3, 0.3, 0.3, 0.3,\
                                              0.3, 0.3, 0.3, 0.3]

L23_thalamicBoutons_dataSpecs['FrameRate'] = [7.6, 7.6, 7.6, 7.6,\
                                              7.6, 7.6, 7.6, 7.6]

# L23_thalamicBoutons_dataSpecs['nFramePerTrial'] = [30, 30, 30, 30,\
#                                                    30, 25, 25, 20]

L23_thalamicBoutons_dataSpecs['nBlankFrames'] = [7, 7, 7, 7,\
                                                 7, 7, 7, 7]

L23_thalamicBoutons_dataSpecs['nStimFrames'] = [23, 23, 23, 23,\
                                                23, 18, 18, 13]


savefilepath = dataDir + 'L23_thalamicBoutons_dataSpecs.hdf'

L23_thalamicBoutons_dataSpecs.to_hdf(savefilepath,key='L23_thalamicBoutons_dataSpecs')


#%% L4 LGN targeted axons: datasets 0 to 5 included

L4_LGN_targeted_axons_dataSpecs = pd.DataFrame()

L4_LGN_targeted_axons_dataSpecs['Date'] = ['20220516','20220516','20220517','20220520',\
                                         '20220520','20220525']

L4_LGN_targeted_axons_dataSpecs['Mouse'] = ['Y78','Y78','Y78','Y78',\
                                          'Y78','Y78']

L4_LGN_targeted_axons_dataSpecs['Depth'] = ['Z280','Z290','Z310','Z360',\
                                          'Z400','Z280']

L4_LGN_targeted_axons_dataSpecs['Sessions'] = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],\
                                             [1,2,3,4,5],[1,2,3,4,5]]

L4_LGN_targeted_axons_dataSpecs['PixelSize'] = [0.3, 0.3, 0.3, 0.3,\
                                              0.3, 0.3]

L4_LGN_targeted_axons_dataSpecs['FrameRate'] = [2.84, 2.84, 2.84, 2.84,\
                                              2.84, 2.84]

# L4_LGN_targeted_axons_dataSpecs['nFramePerTrial'] = [10, 10, 10, 10,\
#                                                    10, 10]

L4_LGN_targeted_axons_dataSpecs['nBlankFrames'] = [5, 5, 5, 5,\
                                                   5, 5]

L4_LGN_targeted_axons_dataSpecs['nStimFrames'] = [5, 5, 5, 5,\
                                                   5, 5]

savefilepath = dataDir + 'L4_LGN_targeted_axons_dataSpecs.hdf'

L4_LGN_targeted_axons_dataSpecs.to_hdf(savefilepath,key='L4_LGN_targeted_axons_dataSpecs')


