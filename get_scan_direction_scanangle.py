# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:35:57 2019

@author: klowell
"""
#%%
############################################################################
# This program reads csvs containing information on individual laser pulses
# and assigns the scan direction -- forward = +1 and backwwards = -1 -- 
# at which the pulse was acquired.
####################### ADD_SCANDIRECT ######################################
# This function adds the scan direction to each row and returns the modified
# dataframe.
def add_scandirect(df):
# Get forward looking rolling average. Because python is a bit stupid, we
# have to reverse the df, calculate the rolling average, and then
# re-reverse the df.
# Add rolling mean as a column to this dataframe. Observation indicated
# that at the edge of a scan, approximately 25-45 pulses will have a scan
# angle of +/-20. A rolling mean of 20 rows should cause the mid-point to
# reverse the scan direction.
    windowsize=20
    df=df[::-1]
    df['roll_mean']=df['scan_angle'].rolling(windowsize,min_periods=2).mean()
    df=df[::-1]
# Set first and last rolling mean to their scan_angle.
    df.loc[0,'roll_mean']=df.loc[0,'scan_angle']
    df.loc[df.shape[0]-1,'roll_mean']=df.loc[df.shape[0]-1,'scan_angle']
# Set initial scan direction by finding first element that is not equal to 
# the first cell. If a segment has a uniform scan angle, its scan direction
# is set equal to zero and its "true" scan direction will be assigned
# later.
    scan_direct=0
    for i in range(1,df.shape[0]):
        if df.loc[i,'roll_mean'] != df.loc[0,'roll_mean']:
            if df.loc[0,'roll_mean'] > df.loc[i,'roll_mean']:
                scan_direct=-1
            else:
                scan_direct=1
            break
# Now loop down this segment and assign scan direction.
    df.loc[0,'scan_direct']=scan_direct
    for i in range(1,df.shape[0]):
# Address change in scan direction. If no change in rolling mean, do
# not change scan direction.
        if df.loc[i,'roll_mean'] > df.loc[i-1,'roll_mean']:
                scan_direct = 1
        elif df.loc[i,'roll_mean'] < df.loc[i-1,'roll_mean']:
                scan_direct = -1
        df.loc[i,'scan_direct']=scan_direct
#    df=df.drop('roll_mean',axis=1)
    return df
########################## TIME_SEGMENTS ####################################
# This function breaks a set of pulses into its components based on
# abnormally large times between pulses. It does
# this by identifying where there are "large" breaks (> 3 std devs) 
# in gpstime between consecutive rows. It returns the starting and ending
# of all segments.
def time_segments(df):
# Prepare to eliliminate the most extreme.
    timediff_mean=df['timediff'].mean()
    timediff_stddev=df['timediff'].std()
    segends=df.index[df['timediff'] >= timediff_mean+ 3.*timediff_stddev]
# Add the first and last row numbers. (Index of df will always start at 0.)
    segends=segends.insert(0,0)
    segends=segends.insert(len(segends),df.shape[0]-1)
    return segends
############################ LOAD LIBRARIES ################################
# Load libraries.
import time
import pandas as pd
import numpy as np
############################# MAIN PROGRAM ##################################
# Set up directories.
############################################################################
in_out_dir='C:/LAS_Kim/LAS_Data/LAS_for_Analysis/'
# I'm a coward: Hardcode some file names. After all, there ARE relatively few....
#file_list=  ['df2016_420500e_2728500n_w_SBET_edge',
#             'df2016_430000e_2707500n_w_SBET_edge',
#                 'df2016_428000e_2719500n_w_SBET_edge',
#                 'df2016_426000e_2708000n_w_SBET_edge']
# I'm a coward: Hardcode some file names. After all, there ARE relatively few....
#file_list=  ['df2016_430000e_2707500n_w_SBET_edge']
file_list=['df2016_426000e_2708000n_w_SBET_edge']
###########################################################################
# Loop through files. Ensure files are sorted on gpstime and properly indexed.
for file in file_list:
    file_tic=time.time()
    print('\nReading file',file,'....')
    dfin=pd.read_csv(in_out_dir+file+'.csv',nrows=100000)
    dfin=dfin.sort_values(by=['gpstime']).reset_index(drop=True)
# Set up a list indicating where flight lines begin and end -- including
# adding a zero at the beginning and an EOF at end.
    dfin['timediff']=dfin['gpstime'].diff().abs()
# Identify start of each flightline by looking for big timediffs between pulses.
# False indicates there will be a single sreening within time_segments.
    flghtends=time_segments(dfin)
#    flghtends=[0,2000]
# Process each flightline.
    for i in range(len(flghtends)-1):
        path_tic=time.time()
        print('\tProcessing flightpath',i+1,'of',len(flghtends)-1)
##########################################################################
#        if i>=1:
#            break
##########################################################################
# Subset each flightline by timediff. timediff in the first row is always problematic
# Assign it a value of 0).
        dftemp=dfin.loc[flghtends[i]:(flghtends[i+1]-1),:].reset_index()
        dftemp.loc[0,'timediff']=0 
# Identify index of all rows with a large timediff between pulses.
# True indicates there will be a single sreening within time_segments.
        scanends=time_segments(dftemp)
        printblock=1
        printrows=100000
# Now process each time break individually.
        for j in range(len(scanends)-1):
# Make dataframe out of each segment and then assign scan direction. If df
# be empty, do not subset and continue (although this would probably occur
# only for the last segment).
            if scanends[j] > scanends[j+1]-1:
                continue
            dfsegment=dftemp.loc[scanends[j]:(scanends[j+1]-1)].reset_index(drop=True)
            dfsegment=add_scandirect(dfsegment)
            if j==0:
                dfallsegs=dfsegment
            else:
                dfallsegs=pd.concat([dfallsegs,dfsegment],axis=0)
#            if j >=5:
#                print(dfkimkim.shape)
            if scanends[j]/printrows>=printblock:
                print('\t\tRows now processed:',scanends[j],'of',dftemp.shape[0])
                printblock += 1
        dfallsegs=dfallsegs.reset_index(drop=True)
# If a segment was uniform, it has no scan direction. If it is in the first
# 90% of the df it the next scan with a scan direction. to avoid going out
# of range, if it is in the last 10%, assign it the most recent one.
# direction of the next segment that has a scan direction of -1 or 1.
        print('\t\tAssigning unassigned scan directions....')
        pct90=int(dfallsegs.shape[0]*0.9)
        pct10=pct90-1
        for j in range(pct90):
# Test if scan direct of "this cell" is zero. If not, continue searching.
            if dfallsegs.loc[j,'scan_direct'] == 0:
# scan_direct=0 found. Look at scan_direct for next cells.  when scan_direct
# != 0 found, assign it to the cell where scan_direct = 0.
                for k in range(j+1,dfallsegs.shape[0]):
                    if dfallsegs.loc[k,'scan_direct'] != 0:
                        dfallsegs.loc[j,'scan_direct']=dfallsegs.loc[k,'scan_direct']
                        break
# Now do the last 10%
        for j in range(dfallsegs.shape[0]-1,pct10,-1):
# Test if scan direct of "this cell" is zero. If not, continue searching.
            if dfallsegs.loc[j,'scan_direct'] == 0:
# scan_direct=0 found. Look at scan_direct for next cells.  when scan_direct
# != 0 found, assign it to the cell where scan_direct = 0.
                for k in range(j-1,0,-1):
                    if dfallsegs.loc[k,'scan_direct'] != 0:
                        dfallsegs.loc[j,'scan_direct']=dfallsegs.loc[k,'scan_direct']
                        break            
        if i == 0:
            dfout=dfallsegs
        else:
            dfout=pd.concat([dfout,dfallsegs],axis=0)
        print('\t\tTime to process this flightpath (mins):',
              round((time.time()-path_tic)/60,1))
    dfout=dfout.drop('index',axis=1)
    dfout=dfout.reset_index(drop=True) 
    print('\tWriting csv for file',file)
    dfout.to_csv(path_or_buf=in_out_dir+file+'_scan_drctn.csv',index=False)
    print('Time to process this file (mins):',
          round((time.time()-file_tic)/60,1))            
