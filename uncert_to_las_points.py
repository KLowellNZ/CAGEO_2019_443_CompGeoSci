# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:35:57 2019

@author: klowell
"""
#%%
############################################################################
# This program reads from already-produced csv files the lidar points
# for a given tile. To each point, it then affixes
#   -- "macro uncertainty" related to a summary of flight path deviation
#   -- "micro uncertainty" related to the closest (in time) flight path deviation
#   -- "SBET micro uncertainty" related to the X, Y, Z, ptich, yaw, and roll
#      deviation for the closest (in time) point
#
# It outputs one csv and one POINT shapefile for each tile read, and
# outputs a single LINE shapefile containing flightpath information for
# all tiles read.
############################ LOAD LIBRARIES ################################
# Load libraries.
import time
import pandas as pd
import glob
# Load spatial libraries. fiona and shapely must be imported even if not
# called explicitly. 
from shapely.geometry import Point, LineString
import fiona
import geopandas as gpd
###################### LINE_GEOPANDAS ####################################
# This function returns a point geo-dataframe that can be written directly
# as a shapefile.
def line_geopandas(df,xx,yy,groupon_list,projection,dftype):
    df['type']=dftype
    df=df[[xx,yy,'flghtpth','tile','type']]
    df=df.reset_index(drop=True)
    linegeom=[Point(xy) for xy in zip(df[xx],df[yy])]
    dfline=gpd.GeoDataFrame(df,geometry=linegeom)
# Group the individual lines and assign the result to the field 'geometry'.
    dfline=dfline.groupby(groupon_list)['geometry'].apply(lambda x:LineString(x.tolist()))
#    dfline['type']=dftype
# Because the previously specified geometry has changed, the field holding
# the geometry must be re-specified.
    dfline=gpd.GeoDataFrame(dfline,geometry='geometry')
    crsdict={'init' :projection}
    dfline.crs=crsdict
# The index must be reset and not dropped to separate geometry from the
# groupby variables.
    dfline=dfline.reset_index()
    return dfline
###################### POINT_GEOPANDAS ####################################
# This function returns a point geo-dataframe that can be written directly
# as a shapefile.
#def point_geopandas(df,xx,yy,flightpath,tile,projection):
def point_geopandas(df,xx,yy,projection):
    df['geometry']=df.apply(lambda x: Point(float(x[xx]),
        float(x[yy])),axis=1)
#    df['flghtpth']=flightpath
#    df['tile']=tile
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
######################### HYPERPARAMETERS ##################################
las_info_dir='C:/LAS_Kim/LAS_Data/LAS_for_Analysis/'
SBET_info_dir='C:/LAS_Kim/LAS_Data/LAS_SBET/SBET_Processed/'
fp_info_dir='C:/LAS_Kim/LAS_Data/LAS_Spatial/LAS_GetEdge/'
edge_info_dir='C:/LAS_Kim/LAS_Data/LAS_Spatial/LAS_GetEdge/LASEdgeData/'
shapeout_dir='C:/LAS_Kim/LAS_Data/LAS_Spatial/LAS_w_uncert/'
SBET_in_file='SBET_Processed_AllFlightpaths.csv'
fp_in_file='edgepoints_summary_w_wind_date_est.csv'
crs='epsg:26917' # Coordinate Reference System
# I'm a coward: Hardcode some file names. After all, there ARE relatively few....
las_file_list=  ['w_flightpath_df2016_420500e_2728500n.csv',
                 'w_flightpath_df2016_428000e_2719500n.csv'] 
#['w_flightpath_df2016_430000e_2707500n.csv',
#           'w_flightpath_df2016_426000e_2708000n.csv',
#           'w_flightpath_df2016_420500e_2728500n.csv',
#           'w_flightpath_df2016_428000e_2719500n.csv']
# Get list of edge data files.  Trim directory off.
edge_file_list=glob.glob(edge_info_dir+'*.csv')
for i in range(len(edge_file_list)):
    edge_file_list[i]=edge_file_list[i].split('\\')[1]
###########################################################################
# Read in SBET data
dfsbet=pd.read_csv(SBET_info_dir+SBET_in_file)
# Read in fligthpath macro information and create a dictionary of uncert info.
dffp=pd.read_csv(fp_info_dir+fp_in_file)
fpmaxdict={}
fpstdevdict={}
fpznumdict={}
fp_list=[]
fp_dict={}
submaxdict={}
substdevdict={}
subznumdict={}
oldtile=dffp.loc[0,'flghtpth'][8:16]
for i in range(dffp.shape[0]):
# Add all flightpaths and uncert info to a sub-dictionary for a single tile.
    if dffp.loc[i,'flghtpth'][8:16] == oldtile:
        submaxdict[int(dffp.loc[i,'flghtpth'][24:25])] = dffp.loc[i,'maxabsdev']
        substdevdict[int(dffp.loc[i,'flghtpth'][24:25])] = dffp.loc[i,'stddevdev']
        subznumdict[int(dffp.loc[i,'flghtpth'][24:25])] = dffp.loc[i,'znumruns']
        fp_list.append(int(dffp.loc[i,'flghtpth'][24:25]))
#        print('fp_list',fp_list,int(dffp.loc[i,'flghtpth'][24:25]))
        continue
# New tile being addressed. Add subdictionaries to master and start new subdict.
    fpmaxdict[oldtile]=submaxdict
    fpstdevdict[oldtile]=substdevdict
    fpznumdict[oldtile]=subznumdict
    fp_dict[oldtile]=fp_list
    oldtile=dffp.loc[i,'flghtpth'][8:16]
    submaxdict={int(dffp.loc[i,'flghtpth'][24:25]) : dffp.loc[i,'maxabsdev']}
    substdevdict={int(dffp.loc[i,'flghtpth'][24:25]) : dffp.loc[i,'stddevdev']}
    subznumdict={int(dffp.loc[i,'flghtpth'][24:25]) : dffp.loc[i,'znumruns']}
    fp_dict[oldtile]=fp_list
    fp_list=[int(dffp.loc[i,'flghtpth'][24:25])]
# Include the last sub-dictionaires in the major dictionary
fpmaxdict[oldtile]=submaxdict
fpstdevdict[oldtile]=substdevdict
fpznumdict[oldtile]=subznumdict
fp_dict[oldtile]=fp_list
# Read and process each tile.
total_tic=time.time()
for i,lasfile in enumerate(las_file_list):
    lasfile_tic=time.time()
    tile=lasfile[28:36]
    print('\nWorking on tile',tile,'\n\tProcessing macro flightpath info...')
    dflas=pd.read_csv(las_info_dir+lasfile)
##################################################
################ SMALL FILE FOR TESTING ##########
#    dflassmall=dflas[0:1000]
#    dflas=pd.concat([dflassmall,dflas[800000:801000]],axis=0)
#    dflas=dflas.reset_index(drop=True)
################################################
# Add file_id to the dataframe for future combining
    dflas['tile']=tile
# Assign the appropriate sub-dictionaries and map the macro uncert columns.
# column that gets progressively updated.
    max_dict=fpmaxdict[tile]
    stdev_dict=fpstdevdict[tile]
    znum_dict=fpznumdict[tile]
    dflas['maxabsdev']=dflas['flghtpth'].map(fpmaxdict[tile])
    dflas['stddevdev']=dflas['flghtpth'].map(fpstdevdict[tile])
    dflas['znumruns']=dflas['flghtpth'].map(fpznumdict[tile])
# Subset the SBET file for efficiency. Then match each point in the LAS
# data with the temporally closest point.
    print('\tProcessing SBET info....')
    dfsbettemp=dfsbet[dfsbet['tile']==tile]
    dfsbettemp=dfsbettemp.reset_index(drop=True)
# Add summary columns and then merge files.
    dfsbettemp.loc[:,'stdXYZ']=dfsbettemp.loc[:,'stdX'] + \
            dfsbettemp.loc[:,'stdY'] + dfsbettemp.loc[:,'stdZ']
    dfsbettemp.loc[:,'stdYwPtRl']=dfsbettemp.loc[:,'stdheading'] + \
            dfsbettemp.loc[:,'stdpitch'] + dfsbettemp.loc[:,'stdroll']
    dfsbettemp.loc[:,'stdTOT']=dfsbettemp.loc[:,'stdXYZ'] + \
            dfsbettemp.loc[:,'stdYwPtRl']
    dfsbetmerge=dfsbettemp[['time','stdXYZ','stdYwPtRl','stdTOT']]
# Merge occurs here. Rename time to SBET_time.
    dflas=pd.merge_asof(dflas, dfsbetmerge,left_on='gpstime',right_on='time',
                        direction='nearest')
    dflas=dflas.rename(columns={'time':'SBET_time'})
    dflas.loc[:,'SBETtmdif']=dflas.loc[:,'gpstime']-dflas.loc[:,'SBET_time']
# Now do the same thing for the deviations. Loop through the flight paths
# for each tile and get "edge deviation" from the appropriate edge file.
#
# The las df will be processed by flight path. First get name of edge file
# for a flight path.
    for k,path in enumerate(fp_dict[tile]):
        print('\tProcessing edge deviation info (Flightpath',path,')....')
        for filename in edge_file_list:
            if tile in filename and 'pth'+str(path) in filename:
                file_path_name=filename
                break
# Read selected columns of the edgefile and subset the las data frame for efficiency.
        dfedge=pd.read_csv(edge_info_dir+file_path_name,
            usecols=['xgrid','ygrid','xactual','yactual','scan_angle',
                     'gpstime','deviation'])
        dfedge=dfedge.rename(columns={'scan_angle':'edgscnangl',
                                     'gpstime':'edge_time'})
        dfedge['flghtpth']=path
        dfedge['tile']=tile
        dflastemp=dflas[(dflas['tile']==tile) & (dflas['flghtpth']==path)]
        dflastemp=pd.merge_asof(dflastemp,dfedge,left_on='gpstime',
                    right_on='edge_time',direction='nearest')
        dflastemp.loc[:,'edgtmdif']=dflastemp.loc[:,'gpstime']-dflastemp.loc[:,'edge_time']
# For first file and flightpath, create a df that contains all edge
# information for subsequent use. Then concatenate flightpaths to it.
        if k==0:
            dflasout=dflastemp
            if i == 0:
                dfalledge=dfedge
            else:
                dfalledge=pd.concat([dfalledge,dfedge],axis=0)
        else:
            dflasout=pd.concat([dflasout,dflastemp],axis=0)
            dfalledge=pd.concat([dfalledge,dfedge],axis=0)
# Clean up and re-order lasfile.
    dflasout=dflasout.drop(['PreSortIndex','xgrid','ygrid','xactual',
                            'yactual','flghtpth_y','tile_y'],axis=1)
    dflasout=dflasout.rename(columns={'flghtpth_x':'flghtpth',
                                     'tile_x':'tile'})
    col_order=['Index','tile','flghtpth','gpstime','x','y','z','class',
              'num_returns','return_no','scan_angle','single','first_of_many',
                  'last','last_of_many','rela_return_num',
              'SBET_time','SBETtmdif','stdXYZ','stdYwPtRl','stdTOT',
              'edge_time','edgtmdif','edgscnangl','deviation',
                  'maxabsdev','stddevdev','znumruns']
    dflasout=dflasout[col_order]
    dflasout=dflasout.reset_index(drop=True)
    dfalledge=dfalledge.reset_index(drop=True)
    out_name=lasfile.replace('w_flightpath_','')
    out_name=out_name.replace('.csv','')
################ DO NOT WRITE FOR DEBUGGING ###################
    dflasout.to_csv(las_info_dir+out_name+'_w_SBET_edge.csv',index=False)
###############################################################
    print('\t.csv file written. Working on points shapefile...')
################ DO NOT WRITE FOR DEBUGGING ###################
    dfgeomgrd=point_geopandas(dflasout,'x','y',crs)
    dfgeomgrd.to_file(shapeout_dir+out_name+'_w_SBET_and_edge_points.shp',driver='ESRI Shapefile')
###############################################################
    print('\t\tPoints shapefile for this tile complete. Get new tile....')
    lasfile_toc=time.time()
    print('***** Elapsed time for tile',tile,' was {:.1f} mins. *****\n'.format((lasfile_toc - lasfile_tic) / 60))
#######################################################################
########## GENERATE ERROR TO NOT PROCESS SHAPEFILES ###################
print(dfkimkim.shape)
#######################################################################
# csvs written; start producing the line shapefiles.
print('\n\n ***** Now working on shapefiles *****')
tile_set=dfalledge['tile'].unique()
# NOTE: type gets appended in the function line_geopandas
groupon_list=['tile','flghtpth','type']
# Get geopandas for sbet using grouping rather than line-by-line processing
# below.
dfline_sbet=line_geopandas(dfsbet,'X','Y',groupon_list,crs,'SBET')
for kk,tile in enumerate(tile_set):
    tile_tic=time.time()
    print('\tProcessing tile',tile,'....')
    for k,path in enumerate(fp_dict[tile]):
        dfsbettemp=dfsbet[(dfsbet['tile']==tile) & (dfsbet['flghtpth']==path)]
        dfedgetemp=dfalledge[(dfalledge['tile']==tile) & (dfalledge['flghtpth']==path)]
# Ensure these are sorted by gpstime.
        dfedgetemp=dfedgetemp.sort_values(by=['edge_time'])
# Calculate average offset.
        avgx_sbet=dfsbettemp['X'].mean()
        avgy_sbet=dfsbettemp['Y'].mean()
        avgx_edge=dfedgetemp['xgrid'].mean()
        avgy_edge=dfedgetemp['ygrid'].mean()
        x_offset=avgx_sbet-avgx_edge
        y_offset=avgy_sbet-avgy_edge
# Determine if line is N-S or E-W and apply apprpriate offset.
        diffx = abs(dfedgetemp['xgrid'].max()-dfedgetemp['xgrid'].min())
        diffy = abs(dfedgetemp['ygrid'].max()-dfedgetemp['ygrid'].min())
        if diffy > diffx: #N-S flight path
            dfedgetemp['xgrid_offset']=dfedgetemp['xgrid']+x_offset
            dfedgetemp['xactl_offset']=dfedgetemp['xactual']+x_offset
            dfedgetemp['ygrid_offset']=dfedgetemp['ygrid']
            dfedgetemp['yactl_offset']=dfedgetemp['yactual']
        else: # E-W flight path
            dfedgetemp['ygrid_offset']=dfedgetemp['ygrid']+y_offset
            dfedgetemp['yactl_offset']=dfedgetemp['yactual']+y_offset
            dfedgetemp['xgrid_offset']=dfedgetemp['xgrid']
            dfedgetemp['xactl_offset']=dfedgetemp['xactual']
# Get a geo df that has the type attached.
        dfline_sbet=line_geopandas(dfsbet,'X','Y',groupon_list,crs,'SBET')
        dfline_edgegridoffset=line_geopandas(dfedgetemp,'xgrid_offset','ygrid_offset',
                                groupon_list,crs,'gridoffset')
        dfline_edgeactloffset=line_geopandas(dfedgetemp,'xactl_offset','yactl_offset',
                                groupon_list,crs,'actloffset')
        dfline_edgegrid=line_geopandas(dfedgetemp,'xgrid','ygrid',
                                groupon_list,crs,'edgegrid')
        dfline_edgeactual=line_geopandas(dfedgetemp,'xactual','yactual',
                                groupon_list,crs,'edgeactl')
# Combine over all dataframes. 
# NOTE: SBET addresses all flight lines so it is appended only once. 
        if kk==0 and k==0:
            dfshape_all=pd.concat([dfline_sbet, dfline_edgegridoffset,
                    dfline_edgeactloffset,dfline_edgegrid,dfline_edgeactual],
                    axis=0)
#            print('Create line shapefile (kk k tile path dfshape_all.shape)',kk,k,tile,path,dfshape_all.shape)
        else:
            dfshape_all=pd.concat([dfshape_all, dfline_edgegridoffset,
                    dfline_edgeactloffset,dfline_edgegrid,dfline_edgeactual],
                    axis=0)
#            print('Add to line shapefile (kk k tile pathdfshape_all.shape)',kk,k,tile,path,dfshape_all.shape)
# Get the points shapefile for this tile and write it out.
#    print('\t\tLine shapefile for this tile complete. Working on points shapefile....')
#    dfpointstemp=dflasout[dflasout['tile']==tile]
#    dfgeomgrd=point_geopandas(dfpointstemp,'x','y',crs)
#    out_name=las_file_list[kk].replace('w_flightpath_','')
#    out_name=out_name.replace('.csv','')
#    dfgeomgrd.to_file(shapeout_dir+out_name+'_w_SBET_and_edge_points.shp',driver='ESRI Shapefile')
#    print('\t\tPoints shapefile for this tile complete. Get new tile....')
    tile_toc=time.time()
    print('\t***** Elapsed time for tile',tile,' was {:.1f} mins. *****\n'.format((tile_toc - tile_tic) / 60))
total_toc=time.time()
print('***** Total time for all tiles was {:.1f} mins. *****\n'.format((total_toc - total_tic) / 60))
dfshape_all.to_file(shapeout_dir+'Allflghtpths'+'_w_SBET_and_edge_lines.shp',driver='ESRI Shapefile')

