# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:05:58 2019

@author: klowell
"""

#%%
##############################################################################
# This is a utility program that drops variables from big csvs so that
# subsequent analysis will be quicker..
########################  MAIN PART OF PROGRAM  ##############################
# Import libraries.
import pandas as pd
################## FILES AND DIRECTORIES ######################################
in_path='C:/LAS_Kim/LAS_Data/LAS_Topography/'
out_path='C:/LAS_Kim/LAS_Data/LAS_for_Analysis/'
#file_list=['df2016_430000e_2707500n']
file_list=['df2016_430000e_2707500n','df2016_426000e_2708000n',
           'df2016_420500e_2728500n','df2016_428000e_2719500n']
infile_suffix='_alltopo_w_inciangle_azimuth_chunked'
outfilesuffix='_all_for_final_analysis.csv'
#################### hyperparameters #################################
vars2drop=['tile','flghtpth','gpstime','x','y','z','scan_angle',
           'SBET_time','SBETtmdif','edge_time','edgtmdif','edgscnangl',
           'SBETtime']
###########################################################################
# Read csv files, drop varialbes, and output.
for file in file_list:
    print('Reading file',file)
    dfin=pd.read_csv(in_path+file+infile_suffix+'.csv')
    dfin.drop(vars2drop,inplace=True,axis=1)
    print('Writing file',file+outfilesuffix,'\n')
    dfin.to_csv(out_path+file+outfilesuffix,index=False)