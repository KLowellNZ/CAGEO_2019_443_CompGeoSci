# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:26:12 2018

@author: klowell
"""

#%%
#############################################################################
# This is a program that combines various programming pieces to:
# -- Tag each pulse with its associated flightpath (and write to a csv)
# -- Extract the pulses with the best and widest scan angle for a 
#    flightpath (and write to a csv)
# from a csv file containing millions
# of lidar pulses. The end of one flightpath and start of another 
# is indicated because the csv files are sorted by gpstime
# and any "big" time gaps -- i.e., anything bigger than +/- mean+3*stddev
# of timediff in consecutive observations -- indicates where a plane turned
# around (typically about 5-12 minutes (300-700 seconds) or where it flew a
# different path for awhile (3.5 hours or so).
#
# Subsequent to running this program, the program getflightpath_edge_summary.py
# should be run. That program will get the very edge of the pulses for a 
# flight path and summarise how crenulated they are.
########################  MAIN PART OF PROGRAM  ####################
# Import libraries and set up lists for looping.
import pandas as pd
inpath='C:/LAS_Kim/LAS_Data/LAS_for_Analysis/'
outpath='C:/LAS_Kim/LAS_Data/LAS_GetEdge/LAS_flghtpths_scanangle/'
############################## Fle names and paths #######################
#filelist=['df2016_420500e_2728500n.csv','df2016_426000e_2708000n.csv',
#          'df2016_428000e_2719500n.csv','df2016_430000e_2707500n.csv']
filelist=['df2016_420500e_2728500n.csv','df2016_426000e_2708000n.csv',
          'df2016_430000e_2707500n.csv']
#filelist=['df2016_430000e_2707500n.csv']
#filelist=['df2016_426000e_2708000n.csv']
#filelist=['df2016_428000e_2719500n.csv']
outfilesuffix='w_flightpath_'
invars=['Index','x','y','z','num_returns','return_no','class','scan_angle',
        'single','first_of_many','last','last_of_many','rela_return_num',
        'gpstime']
outvars=['PreSortIndex','x','y','z','num_returns','return_no','class','scan_angle',
        'single','first_of_many','last','last_of_many','rela_return_num',
        'gpstime','flghtpth']
filedict = {'df2016_430000e_2707500n.csv':'2707500n',
            'df2016_426000e_2708000n.csv':'2708000n',
            'df2016_428000e_2719500n.csv':'2719500n',
            'df2016_420500e_2728500n.csv':'2728500n'}
#######################################################################
############################## Hyper-parameters #######################
# -- skippoints: It was observed that the first few and last points of a new
#                flight path can be junk. skippoints is the number of points
#                that will be ignored at the start and end of a flightpath.
skippoints=10
######################################################################
# Loop through files.
for k,file in enumerate(filelist):
# Read file, rename index, sort on gpstime, and reset index. Calculate
# timediff between consecutive rows.
    print('***** Now working on file',file,'*****\n   File',k+1,'of',len(filelist))
    flghtpth=1
    dfall=pd.read_csv(inpath+file,usecols=invars)
    dfall=dfall.rename(columns={'Index':'PreSortIndex'})
    dfall.sort_values('gpstime',inplace=True)
    dfall.reset_index(inplace=True,drop=True)
    dfall['time_diff']=dfall['gpstime'].diff()
# time_diff for Row 1 is blank -- delete it. Also get rid of anything with
# missing values.
    dfall=dfall.drop(dfall.index[0])
    dfall=dfall.dropna()
    dfall.reset_index(inplace=True,drop=True)
# Calculate mean and stddev of time_diff to be able to identify flight paths.
# Thresh is only upper limit of 3* stddev because all time_diffs are positive.
    avgtimediff=dfall['time_diff'].mean()
    stdevtimediff=dfall['time_diff'].std()
    thresh=avgtimediff+3*stdevtimediff
# Get indices of rows that are new flight lines -- i.e., that exceed thresh.
# Set up flattened list of the limits of each flightline.
    pathstarts=dfall.index[dfall['time_diff']>thresh].tolist()
    pathlimits=[[0],pathstarts,[dfall.shape[0]]]
    pathlimits = [item for sublist in pathlimits for item in sublist]
# If second item on list is small, there is an error at the beginning of
# the file. Get rid of first limit and start getting points at second item.
    if pathlimits[1] < skippoints:
        pathlimits.pop(0)
    flghtpth=1
# Because a pre-sorted file is read,the first point is start of first flightpath.
# Ignore the first skippoints because of potential problems that had been
# observed.
    print('Getting points for flight path 1')
    dfout=dfall.iloc[(pathlimits[0]+skippoints):(pathlimits[1]-skippoints),:]
    dfout.reset_index(inplace=True,drop=True)
    dfout['flghtpth']=flghtpth
# Now subset chunks of pulses representing individual flight paths.
    newstart=pathlimits[1]
    for j in range(1,len(pathlimits)-1):
        flghtpth += 1
        print('Getting points for flight path',flghtpth)
# Set up end of range. Either the start of the next flight line - skippoints,
# or the end of the file - skippoints.
        newend=pathlimits[j+1]
# If the next start is within skippoints of the end of the previous 
# flightpath, there is an error and we want to ignore the "short flightpath"
        if newend-newstart <= skippoints:
            newstart=newend+skippoints
            flghtpth -= 1
            continue
        dftemp=dfall.iloc[newstart+skippoints:newend-skippoints,:]
        dftemp['flghtpth']=flghtpth
        dfout=pd.concat([dfout,dftemp],axis=0)
        newstart=newend+skippoints
    dfout=dfout.drop('time_diff',axis=1)
    dfout=dfout.reset_index(drop=True)
    dfout.to_csv(path_or_buf=inpath+outfilesuffix+file,index=True,index_label='Index')
# dfout contains the information we need to subset flightpaths by scan angle
# and then get edges. Get the appropriate subset.
    for i in range(1,flghtpth+1):
        print ('Now getting edge scan angles for flight path',i,'data set',file)
# Get dfs for this flightpath and +20 and -20 scan_angle. Then use the one
# that is the longest.
        dfuse = dfout[(dfout['scan_angle']==-19) & (dfout['flghtpth']==i)]
        filetail='_scnangl_mnus19.csv'
        dfalt = dfout[(dfout['scan_angle']==19) & (dfout['flghtpth']==i)]
        distuse=((dfuse['x'].max()-dfuse['x'].min())**2+(dfuse['y'].max()-
                   dfuse['y'].min())**2)**0.5
        distalt=((dfalt['x'].max()-dfalt['x'].min())**2+(dfalt['y'].max()-
                   dfalt['y'].min())**2)**0.5
# If there are no points in dfuse/dfalt, their lengths = Nan which is 
# apparently treated as a large number. Therefore set it to zero.
        if dfuse.shape[0] <= 0:
            distuse = 0
        if dfalt.shape[0] <= 0:
            distalt = 0
        if distalt>distuse:
            dfuse=dfalt
            filetail='_scnangl_plus19.csv'        
# It is possible that (for short flight lines?), the (abs(scan_angle) will
# be <19. If so, take the next lowest scan angle with the longest length.
# Because of the possibility that there are scan_angles > abs(19)
# subset points less than 19.
        if dfuse.shape[0]<=0:
            dftemp=dfout[(dfout['flghtpth']==i) & (dfout['scan_angle'].abs() < 19)]
            maxangle=dftemp['scan_angle'].abs().max()
            dfuse=dftemp[dftemp['scan_angle']==(maxangle*(-1))]
            dfalt=dftemp[dftemp['scan_angle']==maxangle]
            filetail='_scnangl_mnus'+str(maxangle)+'.csv'
            distuse=((dfuse['x'].max()-dfuse['x'].min())**2+(dfuse['y'].max()-
                       dfuse['y'].min())**2)**0.5
            distalt=((dfalt['x'].max()-dfalt['x'].min())**2+(dfalt['y'].max()-
                          dfalt['y'].min())**2)**0.5
# If there are no points in dfuse/dfalt, their lengths = Nan which is 
# apparently treated as a large number. Therefore set it to zero.
            if dfuse.shape[0] <= 0:
                distuse = 0
            if dfalt.shape[0] <= 0:
                distalt = 0
            if distalt>distuse:
                dfuse=dfalt
                filetail='_scnangl_plus'+str(maxangle)+'.csv'
        nameout=filedict[file]+'flghtpth'+str(i)+filetail
##################################################################
#        if i==1:
#            break
##################################################################
        dfuse.to_csv(outpath+nameout,index=True,index_label='Index')
        
    