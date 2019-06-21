# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:31:02 2019

@author: klowell
"""

#%%
#############################################################################
# This program reads a csv file that contains "all" las meta data for each
# pulse -- SBET, my deviations, first of many..., ocean floor slope and
# azimuth -- EXCEPT the horizontal angle of the return -- i.e., the azimuth
# from the plane to the return. This file adds that so that hopefully I can
# FINALLY(!!!!!) start analysing data.
# The original formulation took way too long to process pulses. It has 
# been re-structured to increase speed.To further increase speed, 
# roll-pitch-yaw are read from an SBET file in which each flightpath is
# summarised in half-second increments.
#
# THIS FORMULATION FURTHER RESTRUCTURES THE PROGRAM USING CHUNKING.
#
# VERY IMPORTANT NOTE!@!@!@!: If this programs crashes after changes are
# made, it may be because GDAL is doing "unknown things" in memory and
# behind the scenes. For example, during development, not deleting a tiff
# that had been previously created caused problems (i.e., it would not
# overwrite the existing file).  
# !!!!!!!!!!!!!! IF THIS OCCURS, EXIT AND THEN RESTART SPYDER. !!!!!!!!!!!!!
###################### POINT_GEOPANDAS ####################################
# This function returns a point geo-dataframe that can be written directly
# as a shapefile.
def point_geopandas(df,xx,yy,projection):
    df['geometry']=df.apply(lambda x: Point(float(x[xx]),
        float(x[yy])),axis=1)
    dfgeom=gpd.GeoDataFrame(df,geometry='geometry')
    crsdict={'init' :projection}
    dfgeom.crs=crsdict
    return dfgeom
########################  MAIN PART OF PROGRAM  ##############################
# Import libraries.
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint
import math
from bisect import bisect_left, bisect_right
# Elevation is necessary for richdem that produces the slope and aspect tifs.
# fiona is necessary for GDAL functionality.
import elevation, fiona, rasterio
from osgeo import ogr, osr
import gdal
import time
################## FILES AND DIRECTORIES ######################################
SBET_path='C:/LAS_Kim/LAS_Data/LAS_SBET/SBET_Processed/'
in_out_path='C:/LAS_Kim/LAS_Data/LAS_Topography/'
#outfile_prefix='Debug_BathyGrid_270500n_linear'
SBET_file='SBET_Processed_AllFlightpaths_halfsec'
#file_list=['df2016_426000e_2708000n_alltopo']
#file_list=['df2016_420500e_2728500n_alltopo']
#file_list=['df2016_426000e_2708000n_alltopo','df2016_420500e_2728500n_alltopo']
#file_list=['df2016_428000e_2719500n_alltopo']
#file_list=['Test_df2016_430000e_2707500n_alltopo']
file_list=['df2016_430000e_2707500n_alltopo','df2016_426000e_2708000n_alltopo',
           'df2016_420500e_2728500n_alltopo','df2016_428000e_2719500n_alltopo']
outfile_suffix='_w_inciangle_azimuth_chunked'
#################### hyperparameters #################################
# Dictionary for subsetting SBET info for efficiency.
# crs is the coordinate reference system to make a shapefile.
# scannerangle is the angle of the mirror of the (circular) scan.
SBET_list=['2708000n','2728500n','2719500n','2707500n']
crs='epsg:26917'
scannerangle=20
###########################################################################
# Read SBET file with all tile and flightpath info.
dfSBET=pd.read_csv(SBET_path+SBET_file+'.csv')
# Loop through the files. prevrows is used to print total rows processed.
for k,file in enumerate(file_list):
    prevrows=0
#################################################################
#    if k==1:
#       break 
#################################################################
    time_tic=time.time()
# Subset SBET info for the current file to save search time. Sort by time
# (even though it should already be sorted by time). Make list of time.
    for tilename in SBET_list:
        if tilename in file:
            dftile_SBET=dfSBET[dfSBET['tile']==tilename]
    dftile_SBET=dftile_SBET.sort_values(by='time').reset_index(drop=True)
    SBETtimelist=dftile_SBET['time'].tolist()
# Read the file and get the total bounding rectangle.
    print('\n\n***** Reading file',file,'*****')
    chunknumb=0
######################################################################
    for dfin in pd.read_csv(in_out_path+file+'.csv',chunksize=100000):
        dfin=dfin.reset_index(drop=True)
        chunknumb += 1
################ FOR DEBUGGING ##################################
#        if chunknumb >=8:
#            break
#################################################################
# Loop through each pulse and affix the closest 0.5 second heading, pitch,
# and roll.
        for i in range(dfin.shape[0]):
            if (i%50000)==0:
                time_toc=time.time()
                elapsedtime=(time_toc-time_tic)/60
                millrowtime=elapsedtime*1000000/(i+prevrows+1)
                print('Processing chunk {:d} row {:d}  Total elapsed time(mins): {:5.1f}  Million row time: {:5.1f}'.format(chunknumb,i,elapsedtime,millrowtime))
# Find closest SBET record and attach its heading, pitch, and roll to the pulse.
            afterindex=bisect_left(SBETtimelist,dfin.loc[i,'gpstime'])
            beforeindex=afterindex-1
# Do not let index go out of range.
#            afterindex=min(afterindex,len(SBETtimelist)-1)
            if abs(SBETtimelist[beforeindex]-dfin.loc[i,'gpstime']) <= \
               abs(SBETtimelist[afterindex]-dfin.loc[i,'gpstime']):
                idxclosest=beforeindex
            else:
                idxclosest=afterindex
# The following is 10% slower and is not used.
#            idxclosest=abs(dffltpath_SBET['time']-dftemp.loc[i,'gpstime']).idxmin()
            dfin.loc[i,'heading']=dftile_SBET.loc[idxclosest,'heading']
            dfin.loc[i,'pitch']=dftile_SBET.loc[idxclosest,'pitch']
            dfin.loc[i,'roll']=dftile_SBET.loc[idxclosest,'roll']
            dfin.loc[i,'SBETtime']=dftile_SBET.loc[idxclosest,'time']
# This flightpath processed. Set up/attach to output dataframe.
        if chunknumb ==1:
            dfout=dfin
        else:
            dfout=pd.concat([dfout,dfin],axis=0)
        dfout=dfout.reset_index(drop=True)
        prevrows += dfin.shape[0]
        print('\t{:d} rows now have heading, pitch, and roll.'.format(prevrows))
    print('Y-P-R found for all flightpaths. Get pulse azimuth and incidence angle....')
#    dfout.to_csv(in_out_path+'Test'+file+outfile_suffix+'.csv',index=False)
#    print(kimkim.shape)
    angle_time_tic=time.time()
# Heading-pitch-roll assigned to each pulse. Process geographic quadrants
# to get azimuth2pls and vertical angle of incidence. First convert scan angle
# rank to degrees. (180/40) = 4.5
    dfout['exactdegree']=4.5*dfout['scan_angle']
# Subset Quadrant 1
    dfNE=dfout.copy()
#    dfNE=dfout[(dfout['scan_direct']>0) & (dfout['scan_angle']>=0)]
#    dfNE['azim2plse']=dfNE['heading']+dfout['exactdegree']
    dfNE=dfNE[(dfNE['scan_direct']>0) & (dfNE['scan_angle']>=0)]
    dfNE['azim2plse']=dfNE['heading']+dfNE['exactdegree']
    dfNE['inciangle']=scannerangle + dfNE['pitch']*np.cos(np.radians(dfNE['exactdegree'])) - \
        dfNE['roll']*np.sin(np.radians(dfNE['exactdegree']))
# Subset Quadrant 2.
    dfSE=dfout.copy()
#    dfSE=dfout[(dfout['scan_direct']<0) & (dfout['scan_angle']>=0)]
#    dfSE['azim2plse']=dfSE['heading']+180-dfout['exactdegree']
    dfSE=dfSE[(dfSE['scan_direct']<0) & (dfSE['scan_angle']>=0)]
    dfSE['azim2plse']=dfSE['heading']+180-dfSE['exactdegree']
    dfSE['inciangle']=scannerangle + dfSE['pitch']*np.cos(np.radians(180-dfSE['exactdegree'])) - \
        dfSE['roll']*np.sin(np.radians(180-dfSE['exactdegree']))
# Subset Quadrant 3.
    dfSW=dfout.copy()
#    dfSW=dfout[(dfout['scan_direct']<0) & (dfout['scan_angle']<0)]
#    dfSW['azim2plse']=dfSW['heading']+180-dfout['exactdegree']
    dfSW=dfSW[(dfSW['scan_direct']<0) & (dfSW['scan_angle']<0)]
    dfSW['azim2plse']=dfSW['heading']+180-dfSW['exactdegree']
    dfSW['inciangle']=scannerangle + dfSW['pitch']*np.cos(np.radians(180-dfSW['exactdegree'])) - \
        dfSW['roll']*np.sin(np.radians(180-dfSW['exactdegree']))
# Subset Quadrant 4.
    dfNW=dfout.copy()
#    dfNW=dfout[(dfout['scan_direct']>0) & (dfout['scan_angle']<0)]
#    dfNW['azim2plse']=dfNW['heading']+360+dfout['exactdegree']
    dfNW=dfNW[(dfNW['scan_direct']>0) & (dfNW['scan_angle']<0)]
    dfNW['azim2plse']=dfNW['heading']+360+dfNW['exactdegree']
    dfNW['inciangle']=scannerangle + dfNW['pitch']*np.cos(np.radians(360+dfNW['exactdegree'])) - \
        dfNW['roll']*np.sin(np.radians(360+dfNW['exactdegree']))
    print('Angles calculated in {:4.1f} mins. Output csv and shapefile\n\n'.format(time.time()-angle_time_tic))
# Recombine data frames and output
    dfoutfinal=pd.concat([dfNE, dfSE, dfSW, dfNW],axis=0).reset_index(drop=True)
# Make maximum angle 0 to 360 (rather than 0 to 720)
    dfoutfinal['azim2plse']=np.where(dfoutfinal['azim2plse']>360,dfoutfinal['azim2plse']-360,
              dfoutfinal['azim2plse'])
    dfoutfinal.drop('exactdegree',axis=1,inplace=True)
    dfoutfinal.to_csv(in_out_path+file+outfile_suffix+'.csv',index=False)
# Make and output a shapefile.
    dfshape=point_geopandas(dfoutfinal,'x','y',crs)
    dfshape.to_file(in_out_path+file+outfile_suffix+'.shp',driver='ESRI Shapefile')