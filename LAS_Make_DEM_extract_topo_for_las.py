# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:26:12 2018

@author: klowell
"""

#%%
#############################################################################
# THIS PROGRAM IS NOT COMPLETELY DEBUGGED. IT TRIED TO SHOOT LINES FROM
# THE CENTRE OF THE CONVEX HULL TO JUST BEYOND THE EXTERIOR OF THE TILE
# FOR INTERPOLATION.  HOWEVER, THE GEOMETRY BECAME OVERWHELMING. IN PARTICULAR
# THE FUNCTION line_ends WOULD NEED A LOT OF WORK.  THIS HAS BEEN ABANDONED
# IN FAVOUR OF A PROGRAM THAT SETS POINTS ON THE EXTERIOR OF RTILE AND 
# THEN EXTRACTS TOPO BASED ON THOSE POINTS.
#
# This program reads a csv file (that was a LAS file), makes a DEM from
# interpolating within the convex hull of the bathymetry points, extrapolates
# beyond the convex hull,produces a DEM that covers the entire geographic
# extent of the las points, gets slope and aspect, extracts elevation, 
# slope, and aspect for all las points, and outputs the result as a csv.
#
# VERY IMPORTANT NOTE!@!@!@!: If this programs crashes after changes are
# made, it may be because GDAL is doing "unknown things" in memory and
# behind the scenes. For example, during development, not deleting a tiff
# that had been previously created caused problems (i.e., it would not
# overwrite the existing file).  
# !!!!!!!!!!!!!! IF THIS OCCURS, EXIT AND THEN RESTART SPYDER. !!!!!!!!!!!!!
################## BOUNDING_BOX ###########################################
# This function returns xmin, xmax, ymin, ymax of a a dataframe.
def bounding_box(df,xattrib,yattrib,round_off):
    if round_off:
        xmin=int(df['x'].min()) # int always rounds down
        xmax=int(df['x'].max()+1) # ensure we always round up 
        ymin=int(df['y'].min()) # int always rounds down
        ymax=int(df['y'].max()+1) # ensure we always round up
    else:
        xmin=df[xattrib].min()
        xmax=df[xattrib].max()
        ymin=df[yattrib].min()
        ymax=df[yattrib].max()
    return xmin, xmax, ymin, ymax
# rounded up integer.
################## CALC_ASPECT_SLOPE ######################################
# This calculates the slope from a tif file and returns a tif file.
def calc_aspect_slope(DEMname):
    DEM=rd.LoadGDAL(DEMname)
    slopepct=rd.TerrainAttribute(DEM,attrib='slope_percentage')
    aspect=rd.TerrainAttribute(DEM,attrib='aspect')
# to be correct for the shapefile, axes in the array must be reversed and
# the values in each "y-column list" reversed. Also, replace all -9999.99 with nans.
    invslopepct=np.swapaxes(slopepct,0,1)
    invslopepct=np.flip(invslopepct,axis=1)
    invaspect=np.swapaxes(aspect,0,1)
    invaspect=np.flip(invaspect,axis=1)
# Assign missing values to nan (instead of default of -9999.99)
    invslopepct[invslopepct <0] ='nan'
    invaspect[invaspect < 0] ='nan'
    return slopepct,invslopepct,aspect,invaspect
################# CALC_NEWX_NEWY ############################################
# This function will calclate the x,y coordinates for the cutoff for the
# points at the "tip" of each buffer spoke.
def calc_newx_newy(df,xcentre,ycentre,bordersize,angle):
# Get the bounding box of these points; False as argument means no rounding.
    xmin_sub,xmax_sub,ymin_sub,ymax_sub=bounding_box(df,'x','y',False)
# Calculate distance from centre to "upper" bound.
    if angle <= 90:
        hypot=((xmax_sub-xcentre)**2+(ymax_sub-ycentre)**2)**0.5
    elif angle>90 and angle <=180:
        hypot=((xmax_sub-xcentre)**2+(ymin_sub-ycentre)**2)**0.5
    elif angle>180 and angle <270:
        hypot=((xmin_sub-xcentre)**2+(ymin_sub-ycentre)**2)**0.5
    else:
        hypot=((xmin_sub-xcentre)**2+(ymax_sub-ycentre)**2)**0.5
# Subtract bordersize from the length and get x,y coords of the lower limit
# of the sample. As angle increases, number of points decreases. Extending the
# hypotenuse by the cube of the sin turns out to be a good scaler to increase
# sample size.
    if (angle>45 and angle <=135) or (angle>225 and angle <=315):
        newhypot=hypot-bordersize#*(1+abs(math.cos(math.radians(angle))**3))
#        print('\nAangle hypot newhypot diff adj',angle,hypot,newhypot,hypot-newhypot,abs(math.sin(math.radians(angle))))
    else:
        newhypot=hypot-bordersize#*(1+abs(math.sin(math.radians(angle))**3))
#        print('\nBangle hypot newhypot diff adj',angle,hypot,newhypot,hypot-newhypot,abs(math.sin(math.radians(angle))))
    if angle <= 90:
        newx=xcentre+math.sin(math.radians(angle))*newhypot
        newy=ycentre+math.cos(math.radians(angle))*newhypot
    elif angle>90 and angle<180:
        newx=xcentre+math.sin(math.radians(angle))*newhypot
        newy=ycentre+math.cos(math.radians(angle))*newhypot
    elif angle>=180 and angle <270:
        newx=xcentre+math.sin(math.radians(angle))*newhypot
        newy=ycentre+math.cos(math.radians(angle))*newhypot
    else:
        newx=xcentre+math.sin(math.radians(angle))*newhypot
        newy=ycentre+math.cos(math.radians(angle))*newhypot
#    print('angle newx newy hypot newhypot dif adj',angle,newx,newy,hypot,newhypot,hypot-newhypot,abs(math.sin(math.radians(angle))))
    return newx,newy
############## CONVEX_HULL_BUFF ############################################
# This function returns a buffer of specified distance (negative for interior)
# for the convex hull of a set of points.
def convex_hull_buff(df,xx,yy,buffdist,projection):
# Make list of x,y coordinates. Get df of xx and df of yy, bind them, and
# convert to list.
    dfxy=df[[xx,yy]]
    xylist=dfxy.values.tolist()
# Convert to geopandas multipoint
    xymulti=MultiPoint(points=xylist)
    xycnvx=xymulti.convex_hull
    xybuf=xycnvx.buffer(buffdist)
    seriesxy=gpd.GeoSeries(xybuf)
# Convert shapely objects to geopandas. Then set geometry and projection.
    dfbuf=gpd.GeoDataFrame(gpd.GeoSeries(xybuf))
    dfbuf=dfbuf.rename(columns={0:'geometry'}).set_geometry('geometry')
    crsdict={'init' :projection}
    dfbuf.crs=crsdict
    return dfbuf,seriesxy
################### FIT_REG ################################################
# This function fits a regression (to the points at the end of an
# "angle buff") and returns pertinent statistics.
def fit_reg(df,xpred,ypred,angle):
    xylist=[]
    elevlist=[]
    for k in range(df.shape[0]):
        xylist.append([df.loc[k,'x'],df.loc[k,'y']])
        elevlist.append(df.loc[k,'elevintrp'])
    reg=linear_model.LinearRegression()
    reg.fit(xylist,elevlist)
    elevpred=reg.predict(xylist)
    RMSE=mean_squared_error(elevlist,elevpred)**0.5
    r2=r2_score(elevlist,elevpred)
    extrapelev=reg.predict([[xpred,ypred]])[0]
#    print('Angle:{:3d}  Extrap(x,y,z): {:7.2f},{:7.2f},{:7.3f}  n:{:3d}  Rsq:{:5.3f}  RMSE(m):{:7.5f}  Coeffs(a,x,y):{:8.2f} {:8.6f} {:8.6f}'.format(angle,
#          xpred,ypred,extrapelev,df.shape[0],r2,RMSE,reg.intercept_,reg.coef_[0],reg.coef_[1]))
# Make a dataframe of reg stats for future analysis.
    dfreg=pd.DataFrame.from_dict({'0':{'angle':angle,'x':xpred,'y':ypred,
            'extrapz':extrapelev,'n':df.shape[0],'rsqrd':r2,'rmse':RMSE,
            'intrcpt':reg.intercept_,'xcoeff':reg.coef_[0],
            'ycoeff':reg.coef_[1]}},orient='index')
    return extrapelev,dfreg,RMSE
############## GET_POINT_VALUES ############################################
# This function returns a df that has elevation, slope, or aspect 
# added to the output dataframe.
def get_point_values(df,tif_file,attrib):
    gdata=gdal.Open(tif_file)
# Coordinates of tif and df are the same so no Geotransform should be needed.
    gt=gdata.GetGeoTransform()
# Read data and then close.
    data=gdata.ReadAsArray().astype(np.float)
    gdata=None
# Now loop through df coordinates saving each extracted value to a list.
# Transform geographic coordinates into tiff indices.
    val_list=[]
    for i in range(df.shape[0]):
        x=int((df.loc[i,'x']-gt[0])/gt[1])
        y=int((df.loc[i,'y']-gt[3])/gt[5])
# Data are missing for a point if its index does not exist, or its value
# is -9999. Make these missing.
        try:
            if data[y,x] > -999:
                val_list.append(str(data[y,x]))
            else:
                val_list.append('')
        except:
            val_list.append('')
# Values obtained. Now create new column in df.
    df[attrib]=val_list
    return df
####################### LINE_ENDS ########################################
# Function line_ends returns the x,y coordinates of the endpoints of a
# line/wheelspoke centred on the centre of the convex hull. (The trig
# in this is maddening!!!)
def line_ends(i,critangle_ur,critangle_lr,xcentre,ycentre,
              glblxmax,glblxmin,glblymax,glblymin,bordersize):
    if i<=critangle_ur:
        ytemp1=glblymax+bordersize
        ytemp2=glblymin-bordersize
        xtemp1=xcentre+(ytemp1-ycentre)*math.tan(math.radians(i))
        xtemp2=xcentre-(ycentre-ytemp2)*math.tan(math.radians(i+180))
# If either x is too small or large, set new x and solve for y.
        if xtemp1>glblxmax+bordersize:
            xtemp1=glblymax+bordersize
            ytemp1=ycentre+(xtemp1-xcentre)/math.tan(math.radian(i))
        if xtemp2<glblxmin-bordersize:
            xtemp2=glblxmin-bordersize
            ytemp2=ycentre-(xcentre-xtemp2)/math.tan(math.radians(i))
    elif i>critangle_ur and i <= critangle_lr:
        xtemp1=glblxmax+bordersize
        xtemp2=glblxmin-bordersize
        ytemp1=ycentre+(xtemp1-xcentre)/math.tan(math.radians(i)) 
        ytemp2=ycentre-(xcentre-xtemp2)*math.tan(math.radians(270-i))
# If either y is too small or large, set new y and solve for x.
        if ytemp1<glblymin-bordersize:
            ytemp1=glblymin-bordersize
            xtemp1=xcentre+(ycentre-ytemp1)*math.tan(math.radians(180-i))
            
#######################################
        elif ytemp1>glblymax+bordersize:
            ytemp1=glblymax+bordersize
            xtemp1=xcentre+(ytemp1-ycentre)*math.tan(math.radians(i))
#######################################
            
        if ytemp2>glblymax+bordersize:
            ytemp2=glblymax+bordersize
            xtemp2=xcentre-(ytemp2-ycentre)*math.tan(math.radians(180-i))
            
############################################################
        elif ytemp2<glblymin-bordersize:
            ytemp2=glblymin-bordersize
            xtemp2=xcentre+(ycentre-ytemp2)/math.tan(math.radians(270-i))
#############################################################

    else:
        ytemp1=glblymin-bordersize
        ytemp2=glblymax+bordersize
        xtemp1=xcentre+(ycentre-ytemp1)*math.tan(math.radians(180-i))
        xtemp2=xcentre-(ytemp2-ycentre)*math.tan(math.radians(180-i))
# If either x is too small or large, set new x and solve for y.
        if xtemp1>glblxmax+bordersize:
            xtemp1=glblxmax+bordersize
            ytemp1=ycentre-(xtemp1-xcentre)/math.tan(math.radians(180-i))
        if xtemp2<glblxmin-bordersize:
            xtemp2=glblxmin-bordersize
            ytemp2=ycentre+(xcentre-xtemp2)/math.tan(math.radians(180-i))
    return xtemp1,xtemp2,ytemp1,ytemp2
###################### LINE_GEOPANDAS ####################################
# This function returns a line geo-dataframe that can be written directly
# as a shapefile.
def line_geopandas(xlist,ylist,projection):
#    dfline=gpd.GeoDataFrame()
#    dfline['geometry']=dfline.apply(lambda x: LineString([(xlist[0],ylist[0]),
#          (xlist[1],ylist[1])]),axis=1)
#    print('xlist ylist dfline',xlist,'\n',ylist,'\n',dfline)
#    linegeom=[LineString(xy) for xy in zip(xlist,ylist)]
    linegeom=LineString([Point(xlist[0],ylist[0]),Point(xlist[1],ylist[1])])
    dfline=gpd.GeoDataFrame({'line_id':[0]})
    dfline['geometry']=linegeom
# Group the individual lines and assign the result to the field 'geometry'.
#    dfline=dfline.groupby(['line_id'])['geometry'].apply(lambda x:LineString(x.tolist()))
# Because the previously specified geometry has changed, the field holding
# the geometry must be re-specified.
    dfline=gpd.GeoDataFrame(dfline,geometry='geometry')
#    dfline.geometry=dfline['geometry']
    dfline.crs={'init' :projection}
    return dfline
############################ MAKE_GEOTIFF #################################
# This function creates a geotiff raster from a shapefile and writes it to
# disk.
def make_geotiff(filename,source_data,source_layer,srs,xmax,xmin,ymax,ymin,
                 gridspace,xfactor,yfactor):
    x_res = int(((xmax - xmin) / gridspace) + xfactor)
    y_res = int(((ymax - ymin) / gridspace) + yfactor)
    target_data = gdal.GetDriverByName('GTiff').Create(filename, x_res, y_res, 1, gdal.GDT_Float64)
# gridspace/2 offset ensure alignment of tiff cells and grid.
    target_data.SetGeoTransform((xmin-gridspace/2, gridspace, 0,
                                 ymax-gridspace/2, 0, -gridspace))
    target_data.SetProjection(srs.ExportToWkt())
    band = target_data.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
# Rasterize
    gdal.RasterizeLayer(target_data, [1], source_layer, options=['ATTRIBUTE=z'])
# Close the tiff file OR NOTHING WILL APPEAR AND BE SAVED!!!!!
    target_data=None
    band=None
    source_data=None
    source_layer=None
    return
##################### MAKE_SHAPEFILE ######################################
# This function creates a georeferenced shapefile. It returns the shapefile
# and the df created during the process.
def make_shapefile(xgrid,ygrid,topo,crs,attrib):
    dfxyz=pd.DataFrame.from_records({'x':xgrid.flatten(),'y':ygrid.flatten(),
                                    attrib:topo.flatten()})
# Prepare df.
    dfxyz.dropna(inplace=True)
    dfxyz[attrib]=dfxyz[attrib].round(3)
    dfxyz=dfxyz.reset_index(drop=True)
# Get geometry and add it to the shapefile.
# modified.
    dfrast=dfxyz.copy()
    geometry=[Point(xyz) for xyz in zip(dfrast.loc[:,'x'], dfrast.loc[:,'y'],
              dfrast.loc[:,attrib])]
    dfshape=gpd.GeoDataFrame(dfrast,geometry=geometry)
    dfshape.crs={'init':crs}
    dfgeomgrd=point_geopandas(dfrast,'x','y',crs)
    return dfxyz, dfgeomgrd
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
############### READ_ELEV_SHAPEFILE ##########################################
# This function reads a shapefile in a way that allows it to be easily
# rasterised and output as a geotiff.
def read_elev_shapefile(shapefilename,crs):
# First set projection
    srs=osr.SpatialReference()
    srs.ImportFromEPSG(int(crs.split(':')[1]))
# Open the data source and read in the extent and data.
    source_data = ogr.Open(shapefilename)
    source_layer = source_data.GetLayer(0)
    return srs,source_layer,source_data
########################  MAIN PART OF PROGRAM  ##############################
# Import libraries.
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint
import math
# Elevation is necessary for richdem that produces the slope and aspect tifs.
# fiona is necessary for GDAL functionality.
import elevation, fiona, rasterio
from osgeo import ogr, osr
import gdal
import richdem as rd
import time
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
####################################################################
dfin_path='C:/LAS_Kim/LAS_Data/LAS_for_Analysis/'
in_out_path='C:/LAS_Kim/LAS_Data/LAS_Topography/'
#outfile_prefix='Debug_BathyGrid_270500n_linear'
cnvx_elev_suffix='_cnvxelev'
cnvx_slope_suffix='_cnvxpctslope'
cnvx_aspect_suffix='_cnvxaspect'
las_elev_suffix='_las_elev'
las_slope_suffix='_las_slopepct'
las_aspect_suffix='_las_aspect'
cnvx_hull_suffix='_cnvxhull'
file_list=['df2016_426000e_2708000n_w_SBET_edge_scan_drctn_xy.csv']
#file_list= ['df2016_430000e_2707500n_w_SBET_edge_scan_drctn_xy.csv']
#file_list=['df2016_428000e_2719500n_w_SBET_edge_scan_drctn_xy.csv']
#file_list=['df2016_420500e_2728500n_w_SBET_edge_scan_drctn_xy.csv']
#file_list= ['df2016_430000e_2707500n_w_SBET_edge_scan_drctn_xy.csv',
#            'df2016_428000e_2719500n_w_SBET_edge_scan_drctn_xy.csv',
#            'df2016_420500e_2728500n_w_SBET_edge_scan_drctn_xy.csv',
#            'df2016_426000e_2708000n_w_SBET_edge_scan_drctn_xy.csv']
################## USE A SMALL FILE FOR DEBUGGING ####################
#file_list=['df2016_430000e_2707500n_w_SBET_edge_scan_drctn_xy.csv']
#################### hyperparameters #################################
# firstgridspace is the spacing of the grid for the first interpolation.
# gridspace is the distance in m between interpolated points.
# crs is the projection used. (26917 is NAD83 for UTM Zone 17N)
# NoData_value is used when creating tifs, but I do not think it is used.
# bufferwidth is size of buffer around lines for extrapolation. both sides
# of a line are buffered meaning a width of bufferwidth*2 are used.
# bordersize is the number of metres used to subset points for extrapolation
#   and the distance outside the las points bounding box that extrapolation
#   points will be set.
# maxrmse is the maximum rmse for "angle spoke" regressions that will be used.
# vardict is used to rename topo variables with their gridsize.
##### GRID SPACING ##########
firstgridspace=1
gridspace=5
#############################
crs='epsg:26917'
NoData_value = -9999
bufferwidth=2
bordersize=10
maxrmse=0.2
#vardict={'elevintrp':'elevintrp_5m','slopepct':'slopepct_5m','aspect':'aspect_5m'}
###########################################################################
# Loop through the files.
for k,file in enumerate(file_list):
#################################################################
#    if k==1:
#       break
#################################################################
    time_tic=time.time()
###################################################################
    outfile_prefix='Test_'+file.split('_w')[0]+'_5m_grid'
###################################################################
# Read the file and get the total bounding rectangle.
    print('\n\n***** Reading file',file,'*****')
######################################################################
    dfbig=pd.read_csv(dfin_path+file)
# Get bounding box and keep decimals if False else roundoff if True
    glblxmin,glblxmax,glblymin,glblymax=bounding_box(dfbig,'x','y',True)
# If the difference between min and max is not an exact multiple of the
# grid spacing, an extra row and/or column has to be added.
    xfactor=0
    yfactor=0
    if (glblxmax-glblxmin)%gridspace != 0:
        xfactor=1
    if (glblymax-glblymin)%gridspace != 0:
        yfactor=1
    lasxgrid,lasygrid = np.mgrid[
            (glblxmin-gridspace):(glblxmax+gridspace+xfactor):gridspace,
            (glblymin-gridspace):(glblymax+gridspace+yfactor):gridspace]
# Subset only the bathymetry points.
    dfin=dfbig[dfbig['class']=='Bth'].reset_index(drop=True)
# Get the bounding box of bathymetry points.  Keep decimals.
    xmin,xmax,ymin,ymax=bounding_box(dfin,'x','y',True)
# Generate 1 m grid at which interpolation will occur and convert to arrays.
# This was first tried using FOR loops to create nested arrays, but it would
# not work, despite all data and structures seemingly being the same.
# NOTE: To ensure coverage of all bathymetric points, 2 m must be added to
# xmax and ymax and subtracted from subtracted from xmin and ymin.  (Python 
# indexing will add a 1m bufffer)
# row will potentially be cut off.
#
# NOTE: The first interpolation will be done using a 1 m grid to improve
    xgrid, ygrid = np.mgrid[xmin-1:xmax+1:firstgridspace,ymin-1:ymax+1:firstgridspace]
    xytuples=dfin.loc[:,['x','y']].values
    ztuples=(dfin.loc[:,['z']]).values
    ztuples=ztuples.flatten()
# Interpolate within the convex hull.
    dem=griddata(xytuples, ztuples, (xgrid,ygrid), method='linear')
# Eyecandy: Plot the interpolated points. Produce a dataframe and shapefile
# and output the shapefile. (The df is currently not written to csv.)
    plt.figure(figsize=(6,12))
    plt.title('Convex hull elevation, slope, aspect maps')
    plt.imshow(dem.T,extent=(xmin-1,xmax+1,ymin-1,ymax+1),origin='lower',cmap='YlOrBr')
    dfelev, dfelevshape = make_shapefile(xgrid,ygrid,dem,crs,'z')
    dfelevshape.to_file(in_out_path+outfile_prefix+cnvx_elev_suffix+'.shp',driver='ESRI Shapefile')
# Read the elev shapefile that was just written and create the tif.
    point_file = in_out_path+outfile_prefix+cnvx_elev_suffix+'.shp'
    srs,source_layer,source_data=read_elev_shapefile(point_file,crs)
# Create the geotiff from the elev shapefile and write it to disc. First
# specify the filename of the raster Tiff that will be created. The zeroes
# passed are the number of additional columns, rows to add to the grid.
# None for the 1m grid but possible more for the larger grid.
    tif_outfile = in_out_path+outfile_prefix+cnvx_elev_suffix+'.tif'
    make_geotiff(tif_outfile,source_data,source_layer,srs,
                 xmax+1,xmin-1,ymax+1,ymin-1,firstgridspace,0,0)
# Now get slope and aspect using the tif file that was just created and 
# output it as shapefile.
    elev_tif=tif_outfile
# slopepct/aspect are for the tiff; invslopepct/invaspect are for the shapefile.
    slopepct, invslopepct, aspect, invaspect = calc_aspect_slope(elev_tif)
# Eyecandy: Display the slope map and the aspect map.
    rd.rdShow(slopepct,cmap='magma',figsize=(5,10))
    rd.rdShow(aspect,cmap='viridis',figsize=(5,10))
    dfslope, dfslopeshape = make_shapefile(xgrid,ygrid,invslopepct,crs,'slope')
    dfaspect, dfaspectshape = make_shapefile(xgrid,ygrid,invaspect,crs,'aspect')
    rd.SaveGDAL(in_out_path+outfile_prefix+cnvx_slope_suffix+'.tif',slopepct)
    rd.SaveGDAL(in_out_path+outfile_prefix+cnvx_aspect_suffix+'.tif',aspect)
    dfslopeshape.to_file(in_out_path+outfile_prefix+cnvx_slope_suffix+'.shp',driver='ESRI Shapefile')
    dfaspectshape.to_file(in_out_path+outfile_prefix+cnvx_aspect_suffix+'.shp',driver='ESRI Shapefile')
# Extract elevation, slope, and aspect values from tiff and attach to
# points in output dataframe.
    dfin=get_point_values(dfin,tif_outfile,'elevintrp')
    dfin=get_point_values(dfin,in_out_path+outfile_prefix+cnvx_slope_suffix+'.tif',
                         'slopepct')
    dfin=get_point_values(dfin,in_out_path+outfile_prefix+cnvx_aspect_suffix+'.tif',
                          'aspect')
# First interpolation is now completed. Get interior buffer of convex hull,
# centre point of the interior of the convexhull (with decimals), and points
# within the interior buffer.This is available as a Geopandas df and a
# GeoSeries (because intersection requires this). Output convex hull as
# polygonal shapefile.
    dfhullbuf,serieshullbuf=convex_hull_buff(dfelevshape,'x','y',-1*gridspace,crs)
    dfhullbuf.to_file(in_out_path+outfile_prefix+cnvx_hull_suffix+'.shp',driver='ESRI Shapefile')
    dfhullpts=dfelevshape[dfelevshape.geometry.intersects(serieshullbuf[0])].reset_index(drop=True)
    xcentre=dfhullbuf.centroid.x[0]
    ycentre=dfhullbuf.centroid.y[0]
# Critangles are angles from the centre of the convex hull to the four
# corners of the bounding box.
    critangle_ur=math.degrees(math.atan((glblxmax-xcentre)/(glblymax-ycentre)))
    critangle_lr=180-math.degrees(math.atan((glblxmax-xcentre)/(ycentre-glblymin)))
    critangle_ll=180+math.degrees(math.atan((xcentre-glblxmin)/(ycentre-glblymin)))
    critangle_ul=360-math.degrees(math.atan((xcentre-glblxmin)/(glblymax-ycentre)))
# For clarity, rename z values as elevinterp.
    dfhullpts.rename(columns={'z':'elevintrp'},inplace=True)
#    dfhullbuf.to_file(in_out_path+'Debug_testcnvxhullbuffer.shp',driver='ESRI Shapefile')
# Now swing arcs every 10 degrees from centre of bathymetry points.
    print('Extrapolating exterior points....')
    for i in range(0,180,3):
        print('\rWorking on angle [%d] of 180 degrees'%i,end='')
# Get coordinates of both ends of line outside .las points bounding box.
        xtemp1,xtemp2,ytemp1,ytemp2= line_ends(i,critangle_ur,critangle_lr,
            xcentre,ycentre,glblxmax,glblxmin,glblymax,glblymin,bordersize)
        xlist=[xtemp1,xtemp2]
        ylist=[ytemp1,ytemp2]
# Convert this into  geopandas df and create a buffer around the line of
# width 2*bufferwidth (since both sides of line are buffered.
# that will result in a bufferwidth of gridspace*2).
        dfgeoline=line_geopandas(xlist,ylist,crs)
        dfgeobuf=dfgeoline.buffer(bufferwidth)
# Subset points within the convex hull for this angle's buffer. 
        dfpts_subset=dfhullpts[dfhullpts.geometry.intersects(dfgeobuf[0])]
# Convert the GeoSeries dfgeobuf to a geopandas df for output.
        dfgeobuf=gpd.GeoDataFrame(gpd.GeoSeries(dfgeobuf))
        dfgeobuf=dfgeobuf.rename(columns={0:'geometry'}).set_geometry('geometry')        
        dfgeobuf['angle']=i
        dfgeobuf.geometry.name='geometry'
# Write this to a geodataframe that will be output with all buffers.
        if i==0:
            dfgeobuffers=dfgeobuf
        else:
            dfgeobuffers=pd.concat([dfgeobuffers,dfgeobuf],axis=0)
# Get the "limits" that define the tip of an angle buffer.
        newx,newy=calc_newx_newy(dfpts_subset,xcentre,ycentre,bordersize,i)
# Subset points at "tip" of angle buffer and write to geopandas df for output.
        if i<=critangle_ur:
            dfupper=dfpts_subset[(dfpts_subset['y']>=newy)].reset_index(drop=True)
        elif i>critangle_ur and i<= critangle_lr :
            dfupper=dfpts_subset[(dfpts_subset['x']>=newx)].reset_index(drop=True)
        else:
            dfupper=dfpts_subset[(dfpts_subset['y']<=newy)].reset_index(drop=True)
        dfupper['angle']=i 
        if i==0:
            dftippoints=dfupper
        else:
            dftippoints=pd.concat([dftippoints,dfupper],axis=0)
# Create regression to predict the point at the end of the bounding box of
# this angle and get regression stats as a dataframe. Note that this is
# a linear interpolation. Prepare data as lists.
        extrapelev, dfreg, RMSE=fit_reg(dfupper,xtemp1,ytemp1,i)
# If the RMSE for the regression is "excessive" do not use this point.
        if RMSE <= maxrmse:
            if i==0:
                dfregout=dfreg
            else:
                dfregout=pd.concat([dfregout,dfreg],axis=0)
# Add this to the tuples that will be used for the next interpolation that
# provides a full surface.
            xytuples=np.insert(xytuples,len(xytuples),[xtemp1,ytemp1],axis=0)
            ztuples=np.insert(ztuples,len(ztuples),extrapelev,axis=0)
# Now do the same thing for the "lower end" of this angle.
# Get the "limits" that define the tip of an angle buffer.
        newx,newy=calc_newx_newy(dfpts_subset,xcentre,ycentre,bordersize,i+180)
        if i+180<=critangle_ll:
            dflower=dfpts_subset[(dfpts_subset['y']<=newy)].reset_index(drop=True)
        elif i+180>critangle_ll and i+180 <= critangle_ul:
           dflower=dfpts_subset[(dfpts_subset['x']<=newx)].reset_index(drop=True)
        else:
            dflower=dfpts_subset[dfpts_subset['y']>=newy].reset_index(drop=True)
        dflower['angle']=i+180
        dftippoints=pd.concat([dftippoints,dflower],axis=0)
# Create regression to predict the point at the end of the bounding box of
# this angle. Note that this is a linear interpolation. Prepare data as lists.
        extrapelev,dfreg, RMSE=fit_reg(dflower,xtemp2,ytemp2,i+180)
# If RMSE is "excessive" do not use this point.
        if RMSE <= maxrmse:
            dfregout=pd.concat([dfregout,dfreg],axis=0)
# Add this to the tuples that will be used for the next interpolation that
# provides a full surface.
            xytuples=np.insert(xytuples,len(xytuples),[xtemp2,ytemp2],axis=0)
            ztuples=np.insert(ztuples,len(ztuples),extrapelev,axis=0)
# THE FOLLOWING PROVIDES THE OPTION OF WRITING TIP POINTS USED FOR EXTRAPOLATION.
# dfgeobuffers is a GeSeries (but dftippoints is a Geopandas df.) convert to df.
#    dfgeobuffers.reset_index(inplace=True, drop=True)
#    dfgeobuffers=gpd.GeoDataFrame(gpd.GeoSeries(dfgeobuffers))
    dfgeobuffers=dfgeobuffers.rename(columns={0:'geometry'})#.set_geometry('geometry')
    dfgeobuffers.crs={'init':crs}
    dftippoints.crs={'init':crs}
    dfgeobuffers.to_file(in_out_path+outfile_prefix+'_buffers.shp',driver='ESRI Shapefile')
    dftippoints.to_file(in_out_path+outfile_prefix+'_tippoints.shp',driver='ESRI Shapefile')
# Write regression stats to a csv for external analysis.
    dfregout.to_csv(in_out_path+outfile_prefix+'_regstats.csv',index=False)
    print(dfkimkim.shape)
# xytuples/ztuples contains the xy coordinates and elevations of the points
# that will give us complete coverage of the las points surface. Output to
# a shapefile and feed them to the interpolation routine using the global
# grid established earlier.
    print('\nNow producing tifs....')
    gdfpoints=gpd.GeoDataFrame(gpd.GeoSeries(map(Point,xytuples)))
    gdfpoints=gdfpoints.rename(columns={0:'geometry'}).set_geometry('geometry')
    gdfpoints['all_zs']=ztuples
    gdfpoints.crs={'init' :crs}
    gdfpoints.to_file(in_out_path+outfile_prefix+'_edgepoints.shp',driver='ESRI Shapefile')
# Eyecandy: Plot the interpolated points. Produce a dataframe and shapefile
# and output the shapefile. (The df is currently not written to csv.)
    dem=griddata(xytuples, ztuples, (lasxgrid,lasygrid), method='linear')
    plt.figure(figsize=(9,11))
    plt.title('Whole area extrapolated elevation, slope, aspect maps')
    plt.imshow(dem.T,extent=(xmin,xmax,ymin,ymax),origin='lower',cmap='YlOrBr')
    dfelev, dfelevshape = make_shapefile(lasxgrid,lasygrid,dem,crs,'z')
    dfelevshape.to_file(in_out_path+outfile_prefix+las_elev_suffix+'.shp',driver='ESRI Shapefile')
# Now read the elev shapefile that was just written and create the tif.
    point_file = in_out_path+outfile_prefix+las_elev_suffix+'.shp'
    srs,source_layer,source_data=read_elev_shapefile(point_file,crs)
# Create the geotiff from the elev shapefile and write it to disc. First
# specify the filename of the raster Tiff that will be created
    tif_outfile = in_out_path+outfile_prefix+las_elev_suffix+'.tif'
    print('\tWorking on elevation tif....')
# xfactor and/or yfactor = 1 means the difference between min and max is not
# an exact multiple of gridspace and an extra row or column must be added.
    make_geotiff(tif_outfile,source_data,source_layer,srs,
            (glblxmax+gridspace+xfactor),(glblxmin-gridspace),
            (glblymax+gridspace+yfactor),(glblymin-gridspace),
            gridspace,xfactor,yfactor)
#    source_data=None
# Now get slope and aspect using the tif file that was just created and 
# output it as shapefile.
    elev_tif=tif_outfile
# slopepct/aspect are for the tiff; invslopepct/invaspect are for the shapefile.
    slopepct, invslopepct, aspect, invaspect = calc_aspect_slope(elev_tif)
# Eyecandy: Display the slope map and the aspect map.
    rd.rdShow(slopepct,cmap='magma',figsize=(5,10))
    rd.rdShow(aspect,cmap='viridis',figsize=(5,10))
    dfslope, dfslopeshape = make_shapefile(lasxgrid,lasygrid,invslopepct,crs,'slope')
    dfaspect, dfaspectshape = make_shapefile(lasxgrid,lasygrid,invaspect,crs,'aspect')
    print('\tWorking on slope tif....')
    rd.SaveGDAL(in_out_path+outfile_prefix+las_slope_suffix+'.tif',slopepct)
    print('\tWorking on aspect tif....')
    rd.SaveGDAL(in_out_path+outfile_prefix+las_aspect_suffix+'.tif',aspect)
    print('\tWriting shapefiles....')
    dfslopeshape.to_file(in_out_path+outfile_prefix+las_slope_suffix+'.shp',driver='ESRI Shapefile')
    dfaspectshape.to_file(in_out_path+outfile_prefix+las_aspect_suffix+'.shp',driver='ESRI Shapefile')
# Now extract elevation, slope, and aspect values from tiff and attach to
# points in output dataframe.
    dfbig=get_point_values(dfbig,in_out_path+outfile_prefix+las_elev_suffix+
                          '.tif','elevintrp')
    time_toc_elev=time.time()
    print('Elevation overlay done.... Elevation overlay time(mins): {:5.1f}'.format((time_toc_elev-time_tic)/60))
    dfbig=get_point_values(dfbig,in_out_path+outfile_prefix+las_slope_suffix+'.tif',
                         'slopepct')
    time_toc_slope=time.time()
    print('Slope overlay done.... Slope overlay time(mins): {:5.1f}'.format((time_toc_slope-time_toc_elev)/60))
    dfbig=get_point_values(dfbig,in_out_path+outfile_prefix+las_aspect_suffix+'.tif',
                          'aspect')
    time_toc_aspect=time.time()
    print('Aspect overlay done.... Aspect overlay time(mins): {:5.1f}'.format((time_toc_aspect-time_toc_slope)/60))
# Replace empty cells with Nan and then get rid of rows with missing values.
    dfbig=dfbig.mask(dfbig=='')
    origlen=dfbig.shape[0]
#    dfbig=dfbig.dropna(axis=0)
    finallen=dfbig.shape[0]
    print('{:d} rows had missing values and have been dropped.'.format(origlen-finallen))
#    dfin=dfin.drop(['Index'],axis=1)
################## FOR GRIDSIZEs OTHER THAN 1M #################################
# For gridsizes other than 1m keep only the index (for subsequent joining)
# and topovars with name changed to reflect gridsize.
    if gridspace != 1:
        dfbig=dfbig[['Index','elevintrp','slopepct','aspect']]
        dfbig=dfbig.rename(columns=vardict)
################################################################################
    dfbig.to_csv(in_out_path+outfile_prefix+'_alltopo.csv',index=False)
    print('Total time for tile {} was (mins):{:5.1f}'.format(outfile_prefix,(time.time()-time_tic)/60))
#        dfpts_subset.to_file(in_out_path+'Debug_testbufpoints.shp',driver='ESRI Shapefile')
#        if i > 1:
#            dfgeobuf.to_file(in_out_path+'Debug_testbuffer.shp',driver='ESRI Shapefile')
#            dfpts_subset.to_file(in_out_path+'Debug_testbufpoints.shp',driver='ESRI Shapefile')
# Get interpreted elevation Now 
#    dfin.to_csv(in_out_path+outfile_prefix+'_alltopo.csv',index=False)
        
