# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:51:32 2019

@author: klowell
"""

#%%
##############################################################################
# This program reads in a .csv file that has x,y coordinates for points and
# what their "error status" is: TP, TN, FP, or FN. It then breaks the total
# space into cells/pixels, and counts the number of each type of error and
# the total number of points in each cell. It also calculates a number of
# to evaluate the statistical distribution of each cell, and outputs a 
# geotiff of the results so that errors and relative errors can be mapped.
# Finally, it fits a regression between %actual and %expected errors of a
# given type and outputs a scattergram and goodness-of-fit statistics.
############################### DRAW_MAP ####################################
# draw_map provides a colormap of the pixels/cells.
def draw_map(tileid,var,pixel_size,pixels_out):
# Get maximum and minimum values.
    minval=9999
    maxval = -9999
    for pixlist in pixels_out:
        for numb in pixlist:
            if numb == np.nan:
                continue
            if numb < minval:
                minval = numb
            if numb > maxval:
                maxval=numb
# Produce plot.       
    plt.rc('font',size=15)
    fig=plt.figure(figsize=(10,10))
    plt.imshow(pixels_out, cmap='jet')
    plt.title('% of Total Bathy/NotBathy (TP&FN)/(TN&FP) Times 100\n'+
              tileid+'  '+var+' ('+str(pixel_size)+'m pixels)' +
              '\nMax(red) ='+ str(round(maxval,1))+'  Min (blue) =' + str(round(minval,1)))
    plt.colorbar()
    plt.xlabel('Column Number')
    plt.ylabel('Row Number')
    pdfoutfile.savefig(fig)
    fig.clf()
    return
############################# GET_HISTO #####################################
# get_histo tabulates the number of points per cell and attaches to the
# master df the number of points in each cell and the percentage of the
# total times 10,000. It is passed the coordinates of the cells of interest
# as x and y arrays (as well as the bounding box, binsize, and main
# df).If first_call is true, create df; otherwise append.
def get_histo(dfout,bound_box,bins,xarray,yarray,rtrn_name,pct_name,first_call):
    histo,xedges,yedges = np.histogram2d(xarray,yarray,
                    range=bound_box,bins=bins)
# Get coordinates of centre of each cell.
    x_centres=[]
    y_centres=[]
    for i in range(bins):
        x=(xedges[i]+xedges[i+1])/2
        for j in range(bins):
            x_centres.append(x)
            y_centres.append((yedges[j]+yedges[j+1])/2)
# Flatten list of counts.
    flat_counts = [y for x in histo for y in x]
# Now create dataframe if this is first call.
    if first_call:
        histo_dict={'x':x_centres,'y':y_centres,rtrn_name:flat_counts}
        dfout=pd.DataFrame(data=histo_dict)
    else:
# If not first call, create df for appending. This should be the same length
# as the output df because it has the same number of cells.
        histo_dict={rtrn_name:flat_counts}
        dftemp=pd.DataFrame(data=histo_dict)
        dfout=pd.concat([dfout,dftemp], axis=1)
# Get percent of total returns in each cell.
    dfout[pct_name]=(dfout[rtrn_name]/dfout[rtrn_name].sum())*10000
    return dfout
############################## SCATTERPLOT ##################################
# scatterplot creates a scatterplot of two variables.
# x and y are the x and y coordinates in a numpy array, axis_dict provides
    # the axis labels, and pair are the names of the two variables being plotted.
def scatterplot(tileid, x, y, axis_dict, pair, pdfoutfile, xpoints, ypoints,
                regmodel):
# round_dict is used to round to the nearest value to optimise plots.
    round_dict={1:1,2:5.,3:100.,4:500.}
    numcells=str(len(x))
    plt.rc('font',size=15)
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
# get max values and scale to next largest 25.
#    bigmax=max(max(y),max(x))
# Figure out x-interval to have 5 tick marks.
    xinterval=int(max(x)//5)
# Round this interval up according to round_dict
    str_xinterval=str(xinterval)
    xrndintrval=int(math.ceil(xinterval/round_dict[len(str_xinterval)]))*round_dict[len(str_xinterval)]
    # Figure out x-interval to have 5 tick marks.
    yinterval=int(max(y)//5)
# Round this interval up according to round_dict
    str_yinterval=str(yinterval)
    yrndintrval=int(math.ceil(yinterval/round_dict[len(str_yinterval)]))*round_dict[len(str_yinterval)]
# Set the maximum on both axes based on the rounded interval.
    maxxval=xrndintrval*5.25
    maxyval=yrndintrval*5.25
# Scatterplot, 1:1 line, and regression line.
    ax.scatter(x,y, label='Pixels')
# Plotting to the minimum x or y value on both axes provides the 1:1 line.
    ax.plot([0,min(maxxval,maxyval)],[0,min(maxxval,maxyval)],label='1:1 Line',color='red')
# Print regression line and 95% confidence interval. (1 is included because
# the model has an intercept.)
    xregs=[0,maxxval]
    yregs=[regmodel.predict([1,0])[0],regmodel.predict([1,maxxval])[0]]
    ax.plot(xregs,yregs,label='95% CI Regr',color='blue')
# Get 95% CI width using CI on intercept and add to plot.
    CI_width=abs(regmodel.conf_int()[0][0]-regmodel.conf_int()[0][1])
    CIupper=[yregs[0]+CI_width,yregs[1]+CI_width]
    CIlower=[yregs[0]-CI_width,yregs[1]-CI_width]
    ax.plot(xregs,CIupper,color='blue',ls=(0,(5,10)))
    ax.plot(xregs,CIlower,color='blue',ls=(0,(5,10)))
# alpha controls transparency of fill region.
    plt.fill_between(xregs, CIupper, CIlower, color='blue',alpha=0.1)
# Set various plot parameters
    ax.set_xlim([0.0, maxxval])
    ax.set_ylim([0.0, maxyval])
    ax.set_xticks(np.arange(0,maxxval,step=xrndintrval))
    ax.set_yticks(np.arange(0,maxyval,step=yrndintrval))
    ax.set_xlabel(axis_dict[pair[0]])
    ax.set_ylabel(axis_dict[pair[1]])
    xcnt_label=axis_dict[pair[0]].replace('%','')
    ycnt_label=axis_dict[pair[1]].replace('%','')
    ax.set_title('% of Total Points by Type Per Cell Times 100\n'+
        tileid+'  '+axis_dict[pair[0]]+' vs. '+axis_dict[pair[1]]+
        '\n(Pixels='+numcells+'  '+xcnt_label+'='+xpoints+'  '+ycnt_label+'='+ypoints+')'+
        '\n Adj Rsqrd='+str(round(regmodel.rsquared_adj,3))+'  p='+str(round(regmodel.f_pvalue,4))+
                '  RMSE='+str(round(regmodel.mse_resid**0.5,0)))
    ax.legend(loc='upper center')
    pdfoutfile.savefig(fig)
    fig.clf()
    return
############################# MAIN PROGRAM ##################################
# Import libraries.
import pandas as pd
import numpy as np
import math
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
import rasterio
from rasterio.transform import from_origin
#from shapely.geometry import Point
#import fiona
#import geopandas as gpd
#import time
################## PLOT/FIGURE LIBRARIES ###############################
import matplotlib.pyplot as plt
######################## HYPERPARAMETERS ######################################
# pixel_size is the dimension in m of each (square) pixel
pixel_size=20
# crs is the projection -- coordinate reference system.
crs='epsg:26917'
err_type=['TP','TN','FP','FN']
plot_pairs=[['tot','TP'],['tot','TN'],['tot','FP'],['tot','FN'],
            ['Bathy','TP'],['Bathy','FN'],['NotBathy','TN'],['NotBathy','FP']]
# tif_vars are variables for which a "map" will be produced and a geotiff
# file will be output.
tif_vars=['tot_pct','Bathy_pct','TP_pct','Bathy_TPpcts','FN_pct','Bathy_FNpcts',
                    'NotBathy_pct','TN_pct','NotBathy_TNpcts','FP_pct','NotBathy_FPpcts']
# errtype_dict adds better descriptive information.
errtype_dict={'tot':'%All Points','TP':'%True Positives','TN':'%True Negatives',
              'FP':'%False Positives','FN':'%False Negatives',
              'Bathy':'%Bathymetry','NotBathy':'%NotBathy'}
# diff_list contains the information for calculating specific differences.
diff_list=[['Bathy_TPpcts','Bathy_pct','TP_pct'],['Bathy_FNpcts','Bathy_pct','FN_pct'],
           ['NotBathy_TNpcts','NotBathy_pct','TN_pct'],['NotBathy_FPpcts','NotBathy_pct','FP_pct']]
################## FILES AND DIRECTORIES ######################################
#in_out_path='C:/LAS_Kim/LAS_Data/LAS_for_Analysis/'
inpath='C:/LAS_Kim/LAS_Reporting/LAS_Articles/LIDARPulse/LIDAR_Pulse_RipleysK/'
#file_list= ['FinalAnalysis15_XGB_IndividTiles_MainEffects_notopo_df2016_430000e_2707500nRipleysK']
file_list= ['FinalAnalysis15_XGB_IndividTiles_MainEffects_notopo_CLIPPED_df2016_428000e_2719500nRipleysK_CLIPPED']
#file_list= ['FinalAnalysis15_XGB_IndividTiles_MainEffects_notopo_df2016_430000e_2707500nRipleysK',
#            'FinalAnalysis15_XGB_IndividTiles_MainEffects_notopo_df2016_420500e_2728500nRipleysK',
#            'FinalAnalysis15_XGB_IndividTiles_MainEffects_notopo_df2016_428000e_2719500nRipleysK',
#            'FinalAnalysis15_XGB_IndividTiles_MainEffects_notopo_df2016_426000e_2708000nRipleysK']
###############################################################################
# Loop through files
for file in file_list:
    outpath=inpath+file.replace('_IndividTiles_MainEffects_notopo','')+'/'
#    outpath=inpath+file.replace('_IndividTiles_MainEffects_notopo_CLIPPED','')+'/'
# Get compact tile identifier.
#    tileid=(file.split('_')[6]+file.split('_')[7]).replace('RipleysK','')
########################### FOR CLIPPED #############################
    tileid=(file.split('_')[7]+file.split('_')[8]).replace('RipleysK','')
# .csv file, pdf (for graphics), and txt (for fitted models) files.
    csvoutfile_name=outpath+'GridErrorsData_'+file[67:]+'.csv'
    pdfoutfile_name=outpath+'GridErrorsGraphs_'+file[67:]+'.pdf'
    pdfoutfile=PdfPages(pdfoutfile_name)
    regroutfile_name=outpath+'GridErrorsModels_'+file[67:]+'.txt'
    regroutfile=open(regroutfile_name,'w')
    print('Reading file',file)
    dfin=pd.read_csv(inpath+file+'.csv')
# Get lower left coordinates and then establish UR so we always deal with a 
# 500m-by-500m space.
    xmin=dfin['x'].min()
    ymin=dfin['y'].min()
    xmax=xmin+500
    ymax=ymin+500
    bound_box=[[xmin,xmax],[ymin,ymax]]
# Now get histograms to tabulate number of pulses by x and y variables.
    bins=int(500/pixel_size)
# Get cell frequency for all points. Convert x and y to coordinate arrays.
    xarray=dfin['x'].values
    yarray=dfin['y'].values
# Get histogram. The final argument ('True') indicates this call to get_histo
# will create the output df. Subsequent calls will append to the output df.
# Create a dummy dfout for the first time.
    dfout=pd.DataFrame(columns=['x','y','tot_rtrns'])
    dfout=get_histo(dfout,bound_box,bins,xarray,yarray,'tot_rtrns','tot_pct',
                    True)
# Cell frequency for Bathy. Eliminate pixels that have no NotBathy.
    dftype=dfin[(dfin['errtype']=='FN')|(dfin['errtype']=='TP')]
    xarray=dftype['x'].values
    yarray=dftype['y'].values
    error='Bathy'
    totname=error+'_rtrns'
    pctname=error+'_pct'        
    dfout=get_histo(dfout,bound_box,bins,xarray,yarray,totname,pctname,
                False)
# Cell frequency for NotBathy. Eliminate pixels that have no NotBathy.
    dftype=dfin[(dfin['errtype']=='TN')|(dfin['errtype']=='FP')]
    xarray=dftype['x'].values
    yarray=dftype['y'].values
    error='NotBathy'
    totname=error+'_rtrns'
    pctname=error+'_pct'        
    dfout=get_histo(dfout,bound_box,bins,xarray,yarray,totname,pctname,
                False)   
# Cell frequency for individual error types.
    for error in err_type:
        dftype=dfin[dfin['errtype']==error]
        xarray=dftype['x'].values
        yarray=dftype['y'].values
        totname=error+'_rtrns'
        pctname=error+'_pct'        
        dfout=get_histo(dfout,bound_box,bins,xarray,yarray,totname,pctname,
                    False)
# Calculate difference variables.
    for diff in diff_list:
        dfout[diff[0]]=dfout[diff[1]]-dfout[diff[2]]
# Eliminate all cells that have no pulse returns and output the csv.
# First make a copy so the geotiff will be correct.
    dftif=dfout.copy()
    dfout=dfout[dfout['tot_rtrns']>0]
# Output .csv
    dfout.to_csv(csvoutfile_name, index=False)
# Plot variables desired as scatterplot.
    for pair in plot_pairs:
# Eliminate pixels that have fewer than 5 pulse returns forthe x-axis variable in it.
        dftemp=dfout[dfout[pair[0]+'_rtrns']>5]
        xpoints=str(int(dftemp[pair[0]+'_rtrns'].sum()))
        ypoints=str(int(dftemp[pair[1]+'_rtrns'].sum()))
        xarray=dftemp[pair[0]+'_pct'].values
        yarray=dftemp[pair[1]+'_pct'].values
# Fit a regression between x and y to determine if spatial dbns are the same.
# Create column of ones to have a constant.
        xregarray=sm.add_constant(xarray)
        regmodel_form=sm.OLS(yarray,xregarray)
        regmodel=regmodel_form.fit()
# Write to text file.
        print('********** REGRESSION MODEL',file[67:83],'**********\n    (X:',
              errtype_dict[pair[0]],'  Y:',errtype_dict[pair[1]],')',file=regroutfile)
        print(regmodel.summary(),'\n\n',file=regroutfile)
# Print scatterplot with descriptives.
        scatterplot(tileid,xarray,yarray,errtype_dict,pair,pdfoutfile,
                    xpoints,ypoints,regmodel)
# Now produce geotiffs for the percentages for each variable of interest.
# Set up list to get affine transformation and get the transformation.
    affine_transform=from_origin(xmin,ymax,pixel_size,pixel_size)
    for var in tif_vars:
# Format the variable of interest as a nested numpy array/list.
        pixels_out=[]
        for j in range(bins,0,-1):
            temp_list=[]
            for l in range(bins):
                if dftif.loc[(j-1)+(l*25),var] == 0 or \
                         dftif.loc[(j-1)+(l*25),var.split('_')[0]+'_rtrns'] <=5:
                    temp_list.append(np.NaN)
                else:
                    temp_list.append(dftif.loc[(j-1)+(l*25),var])
            pixels_out.insert(0,temp_list)
# Now flip the list so that the geometry comes out right.
        pixels_out.reverse()
# Draw a map.
        draw_map(tileid,var,pixel_size,pixels_out)
# Set up the write file for a 1-band image.
        tifoutfile_name=outpath+var+'_'+file[67:]+'.tif'
# Data have been tabulated by bins of a set size. Use the df that has data
# for all cells without having empty cells eliminated.
        with rasterio.open(tifoutfile_name,'w',driver='GTiff', height=bins,
                width=bins, dtype=dftif[var].dtype, count=1, crs=crs,
#                width=bins, dtype=object, count=1, crs=crs,
                transform=affine_transform) as dest:
            dest.write(pixels_out,1)
        dest.close()
    pdfoutfile.close()
# Make various plots.

