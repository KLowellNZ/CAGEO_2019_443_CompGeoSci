# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:25:46 2019

@author: klowell
"""

#%%
#############################################################################
# This is a utility program to join two DEMs.  These will have been produced
# by the program LAS_Make_DEM_extract_topo_for_las_outside_in.py at two
# different spatial resolutions (nominally 1m and 5m grids). The "1m" DEM
# will contain all information -- e.g., gpstime, scan_angle, scan_direct... --
# but the "5m" file will contain only the common field ("Index"), and the
# elevation, slope, and aspect from the 5m grid.
########################  MAIN PART OF PROGRAM  ##############################
# Import libraries.
import pandas as pd
import numpy as np
import time
####################################################################
in_out_path='C:/LAS_Kim/LAS_Data/LAS_Topography/'
file_pairs=[['df2016_430000e_2707500n_1m_grid_alltopo.csv',
             'df2016_430000e_2707500n_5m_grid_alltopo.csv'],
            ['df2016_428000e_2719500n_1m_grid_alltopo.csv',
             'df2016_428000e_2719500n_5m_grid_alltopo.csv'],
            ['df2016_420500e_2728500n_1m_grid_alltopo.csv',
             'df2016_420500e_2728500n_5m_grid_alltopo.csv'],
            ['df2016_426000e_2708000n_1m_grid_alltopo.csv',
             'df2016_426000e_2708000n_5m_grid_alltopo.csv']]
#file_pairs=[['Test_df2016_430000e_2707500n_1m_grid_alltopo.csv',
#             'Test_df2016_430000e_2707500n_5m_grid_alltopo.csv']]
tot_time_tic=time.time()
for pair in file_pairs:
    file_tic=time.time()
# Set up output file name.
    titlesegments=pair[0].split('_')
    outfilename=''
    for segment in titlesegments:
        if 'm' in segment or 'grid' in segment:
            continue
        if 'csv' in segment:
            outfilename += segment
        else:
            outfilename += segment+'_'
# Read the two DEMs.
    print('Reading 1m file',pair[0],'....')
    df_1m = pd.read_csv(in_out_path+pair[0])
    print('Reading 5m file',pair[1],'....')
    df_5m = pd.read_csv(in_out_path+pair[1])
    print('Rows in 1m df:',df_1m.shape[0],'\nRows in 5m df:',df_5m.shape[0],
          '\nNow joining....')
# Join them on the common column which should only be Index. (Joining
# explicitly on Index kept both Index columns with one being unnamed.)
    dfall=pd.merge(left=df_1m, right=df_5m, how='outer', on='Index')
#    dfall=pd.merge(left=df_1m, right=df_5m, left_on='Index', right_on='Index',
#                   how='outer')
    print('Rows in joined df:',dfall.shape[0],'\nOutputting to csv....')
    dfall.to_csv(in_out_path+outfilename,index=False)
    print('Total time for',outfilename,'in mins:',round((time.time()-file_tic)/60,1))
print('\nTotal program time in mins:',round((time.time()-tot_time_tic)/60,1))
    
    