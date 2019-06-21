# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:35:57 2019

@author: klowell
"""
#%%
############################################################################
# This program reads an SBET file that provides information about the
# flightpath of .las data. It was originally provided to K. Lowell by B. Calder
#  "the code that the TPU estimator uses to load the file (using Pandas)..."
# attached to an email dated 8 January 2019 2:57PM.  It has been
# modified by K. Lowell from its original form.
# It outputs 
#  1)a csv, point shapefile, and a line shapefile containing
#    all relevant SBET information for the 21 flightpaths on the four
#    tiles examined.
#  2)a csv that is a summary by half seconds of all pulses to be used
#    for subsequent processing where memory is an issue and high temporal
#    resolution is unnecessary.
############################ LOAD LIBRARIES ################################
# Load libraries.
import os
import time
import pandas as pd
from datetime import datetime
# Load spatial libraries. fiona and shapely must be imported even if not
# called explicitly. 
from shapely.geometry import Point, LineString, MultiPoint
import fiona
import geopandas as gpd
###################### BUILD_SBETS_DATA ####################################
# Build a single pandas dataframe from all ASCII sbet files
def build_sbets_data(sbet_files):
    sbets_df = pd.DataFrame()
    header_sbet = ['time', 'lon', 'lat', 'X', 'Y', 'Z', 'roll', 'pitch', 'heading',
                   'stdX', 'stdY', 'stdZ', 'stdroll', 'stdpitch', 'stdheading']
    print('\nGetting sbet data from: ')
    for sbet in sorted(sbet_files):
        print('\t{}...'.format(sbet))
        sbet_df = pd.read_table(sbet,skip_blank_lines=True,engine='c',
                  delim_whitespace=True,header=None,names=header_sbet,
                  index_col=False)
        print('\t\t({} trajectory points)'.format(sbet_df.shape[0]))
        sbet_date = get_sbet_date(sbet)
        gps_time_adj = gps_sow_to_gps_adj(sbet_date, sbet_df['time'])
        sbet_df['time'] = gps_time_adj
        sbets_df = sbets_df.append(sbet_df, ignore_index=True)
    sbets_data = sbets_df.sort_values(['time'], ascending=[1])
    return sbets_data
###################### FLIGHTPATH_TIMESUMMARY ###############################
# This function receives SBET information for a flightpath (usually one row
# per 1/200 seconds) and summarises it using averaging by the user-specified
# time interval)
def flightpath_timesummary(df,timesummary,SBET_cols):
#    df=df.reset_index(drop=True)
    timestart=df.loc[0,'time']
    lasttime=df.loc[df.shape[0]-1,'time']
    while timestart<lasttime:
# Subset df.
        dftemp=df[(df['time'] >= timestart) & (df['time']< timestart+timesummary)].reset_index(drop=True)
        flghtpth=dftemp.loc[0,'flghtpth']
        tile=dftemp.loc[0,'tile']
        dftemp=dftemp.drop(['flghtpth','tile'],axis=1)
        dftempmean=pd.DataFrame(dftemp.mean()).transpose()
        dftempmean['flghtpth']=flghtpth
        dftempmean['tile']=tile
        if timestart==df.loc[0,'time']:
            dfreturn=dftempmean
        else:
            dfreturn=pd.concat([dfreturn,dftempmean],axis=0)
        timestart=timestart+timesummary
    dfreturn=dfreturn.reset_index(drop=True)
    return dfreturn
###################### GET_SBET_DATE #######################################
# This function gets the sbet date from the file name that is passed
# to the function as a string.
def get_sbet_date(sbet):
# Strip directory info from filename.
    sbet_name = os.path.basename(sbet)
    year = int(sbet_name[0:4])
    month = int(sbet_name[4:6])
    day = int(sbet_name[6:8])
    sbet_date = [year, month, day]
    return sbet_date
################### GPS_SOW_TO_GPS_ADJ ####################################
# This function converts the GPS seconds-of-week timestamps
# to GPS adjusted standard time
def gps_sow_to_gps_adj(gps_date, gps_wk_sec):
    print('\tConverting GPS week seconds to GPS adjusted standard time...'),
# Set up constants, then create a datetime object that is sbet_date
    SECS_PER_GPS_WK = 7 * 24 * 60 * 60  # 604800 sec
    SECS_PER_DAY = 24 * 60 * 60  # 86400 sec
    GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
    year = gps_date[0]
    month = gps_date[1]
    day = gps_date[2]
    sbet_date = datetime(year, month, day)
    dt = sbet_date - GPS_EPOCH
    gps_wk = int((dt.days * SECS_PER_DAY + dt.seconds) / SECS_PER_GPS_WK)
    gps_time = gps_wk * SECS_PER_GPS_WK + gps_wk_sec
    gps_time_adj = gps_time - 1e9
    return gps_time_adj
###################### LINE_GEOPANDAS ####################################
# This function returns a point geo-dataframe that can be written directly
# as a shapefile.
def line_geopandas(df,xx,yy,groupon_list,projection):
    df=df[[xx,yy,'flghtpth','tile']]
    df=df.reset_index(drop=True)
    linegeom=[Point(xy) for xy in zip(df[xx],df[yy])]
    dfline=gpd.GeoDataFrame(df,geometry=linegeom)
# Group the individual lines and assign the result to the field 'geometry'.
    dfline=dfline.groupby(groupon_list)['geometry'].apply(lambda x:LineString(x.tolist()))
# Because the previously specified geometry has changed, the field holding
# the geometry must be re-specified.
    dfline=gpd.GeoDataFrame(dfline,geometry='geometry')
    crsdict={'init' :projection}
    dfline.crs=crsdict
# the index must be reset and not dropped so that the groupby variables
# become variables/columns rather than indices.
    dfline=dfline.reset_index()
    return dfline
############################# function MAIN ###################################
# Though formulated as a function, this is the main part of the program.
def main(sbet_files):
    sbets_df = build_sbets_data(sbet_files)
    sbet_toc=time.time()
    print('\tIt took {:.1f} mins to load files in the sbets directory.'.format((sbet_toc - sbet_tic) / 60))
    return sbets_df
###################### POINT_GEOPANDAS ####################################
# This function returns a point geo-dataframe that can be written directly
# as a shapefile.
def point_geopandas(df,xx,yy,flightpath,tile,projection):
    df['geometry']=df.apply(lambda x: Point(float(x[xx]),
        float(x[yy])),axis=1)
    df['flghtpth']=flightpath
    df['tile']=tile
    dfgeom=gpd.GeoDataFrame(df,geometry='geometry')
    crsdict={'init' :projection}
    dfgeom.crs=crsdict
    return dfgeom
############### tominutes #######################
# The time required to fit and test each NN is recorded. Function tominutes
# converts time in seconds to hours, minutes, and seconds
def tominutes(timeinseconds):
#    minutes, seconds = divmod(timeinseconds, 60)
#    hours, minutes = divmod(minutes, 60)
    hours=int(timeinseconds/3600)
    minutes=int((timeinseconds- hours*3600)/60)
    seconds=int(timeinseconds-hours*3600-minutes*60)
    return hours,minutes,seconds
############################# MAIN PROGRAM ##################################
# This is the code that runs the program. The directory containing the 
# SBET files must be provided.
#sbet_dir='C:/LAS_Kim/LAS_Data/LAS_SBET/SBET_Raw/'
######################### HYPERPARAMETERS ##################################
sbet_in_dir='C:/LAS_Kim/LAS_Data/LAS_SBET/SBET_Raw/'
sbet_out_dir='C:/LAS_Kim/LAS_Data/LAS_SBET/SBET_Processed/'
outfile='SBET_Processed_AllFlightpaths'
outfile_halfsec=outfile+'_halfsec'
fpinfo_dir='C:/LAS_Kim/LAS_Data/LAS_Spatial/LAS_GetEdge/'
fpfile='edgepoints_summary_w_wind_date_est.csv'
shapeout_dir='C:/LAS_Kim/LAS_Data/LAS_SBET/SBET_Processed/Spatial/'
# in_files=['20160425.1_880_p_tpu_sbet','20160425.2_880_p_tpu_sbet']
crs='epsg:26917' # Coordinate Reference System
# I'm a coward: Hardcode SBET file names. After all, there ARE only 6....
SBET_files=['20160425.1_880_p_tpu_sbet.txt','20160425.2_880_p_tpu_sbet.txt',
            '20160422_880_p_tpu_sbet.txt','20160419_880_p_tpu_sbet.txt',
            '20160423_880_p_tpu_sbet.txt','20160424_880_p_tpu_sbet.txt']
SBET_cols=['time', 'lon', 'lat', 'X', 'Y', 'Z', 'roll', 'pitch', 'heading',
           'stdX', 'stdY', 'stdZ', 'stdroll', 'stdpitch', 'stdheading']
######################### HYPERPARAMETERS ##################################
# endslop is the total excess time wanted on the end of each flight line --
#   i.e., endslop of 10 gives 5 seconds on the end of each line.
# time summary is the interval in seconds for summarising data to decrease
#   temporal resolution and file size.
endslop=10
timesummary =0.5
##### ######################################################################
# Read flight path info and set up a df with the necessary info. (Extend
# the time range by 10 seconds -- 5 each at the beginning and end) --
# to ensure we have sufficient length for subsequent processing.
# Read flight path info and set up a df with the necessary info. (Extend
# the time range by 10 seconds -- 5 each at the beginning and end) --
# to ensure we have sufficient length for subsequent processing.  
# 
# (Though 5 seconds is a lot of excess for some flight lines, it is not for 
# others. In addition, it will be condensed for certain purposes.)
dffp=pd.read_csv(fpinfo_dir+fpfile)
fpcols=['file_id','flghtpth','starttime','endtime']
dffpnew=pd.DataFrame(columns=fpcols)
for i in range(dffp.shape[0]):
    fpdict={'file_id':[dffp.loc[i,'flghtpth'][8:16]],
            'flghtpth':[dffp.loc[i,'flghtpth'][24:25]],
            'starttime':[dffp.loc[i,'starttime']-endslop/2],
            'endtime':[dffp.loc[i,'endtime']+endslop/2]}
    dftemp=pd.DataFrame(data=fpdict)
    dffpnew=dffpnew.append(dftemp)
dffpnew=dffpnew.reset_index(drop=True)
# Loop through each flight path looking for the portions of the SBET files
# that matches each. Combine all SBET data into a single file, and output 
# a point and a line shapefile where each line is a separate shapefile
# object. Also prepare to output the time-based summary.
dfout=pd.DataFrame(columns=SBET_cols)
dfhalfsec=pd.DataFrame(columns=SBET_cols)
# Start search for new flightline.
for i in range(len(dffpnew)):
    sbet_tic=time.time()
###########################################
################ FOR TESTING ##############
#    if i >= 2:
#        break
###########################################
    flght_file=dffpnew.loc[i,'file_id']
    flightpath=dffpnew.loc[i,'flghtpth']
    starttime=dffpnew.loc[i,'starttime']
    endtime=dffpnew.loc[i,'endtime']
# Search each of the SBET files. (Yes this is inefficient because for
# each flight line we will loop through all of the SBET files until a 
# match is found. This has been improved by ordering the search order
# to start with the files known to have the most flight lines.)
    for file in SBET_files:
# Make the filename a list because this is what the function MAIN expects.
        file_list=[sbet_in_dir+file]
# main is a function that manages the original sbet code.
        dfsbet=main(file_list)
# Get the observations for the flightpath of interest. If the df is not empty
# SBET points for this flight line have been found.
        dfflghtpth=dfsbet[(dfsbet['time']>=starttime) & (dfsbet['time']<=endtime)]
        if dfflghtpth.shape[0] > 0:
            dfflghtpth['flghtpth']=flightpath
            dfflghtpth['tile']=flght_file
            dfout=dfout.append(dfflghtpth)
            print ('\t\tFor',flght_file,'Path',flightpath,dfflghtpth.shape[0],'points were found in',file_list[0])
# Now summarise this flightpath by the time interval specified.
            dftemp=dfflghtpth.copy().reset_index(drop=True)
            dftempsec=flightpath_timesummary(dftemp,timesummary,SBET_cols)
            dfhalfsec=pd.concat([dfhalfsec,dftempsec],axis=0)
# Create/add to (multi-)point and line shapefiles.
            groupon_list=['flghtpth','tile']
            if i == 0:
                dfgeomgrd=point_geopandas(dfflghtpth,'X','Y',flightpath,flght_file,crs)
                dfline=line_geopandas(dfflghtpth,'X','Y',groupon_list,crs)
            else:
                dfgeomtemp=point_geopandas(dfflghtpth,'X','Y',flightpath,flght_file,crs)
                dfgeomgrd=pd.concat([dfgeomgrd,dfgeomtemp])
                dfgeomgrd.geometry=dfgeomgrd['geometry']
                dflinetemp=line_geopandas(dfflghtpth,'X','Y',groupon_list,crs)
                dfline=pd.concat([dfline,dflinetemp])
# Flightline found. Break loop to start searching for new flightline.
            sbet_toc=time.time()
            print('\t***** Elapsed time for this flight line: {:.1f} mins. *****\n'.format((sbet_toc - sbet_tic) / 60))
            break
# Write to disk csv and shapefile.
dfout.to_csv(sbet_out_dir+outfile+'.csv',index=False)
dfhalfsec.to_csv(sbet_out_dir+outfile_halfsec+'.csv',index=False)
line_suffix='_line.shp'
dfline.to_file(shapeout_dir+outfile+line_suffix,driver='ESRI Shapefile')
gp_suffix='_points.shp'
dfgeomgrd.to_file(shapeout_dir+outfile+gp_suffix,driver='ESRI Shapefile')
