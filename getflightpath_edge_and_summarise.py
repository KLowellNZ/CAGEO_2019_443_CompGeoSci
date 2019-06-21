# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:26:12 2018

@author: klowell
"""

#%%
#############################################################################
# This program gets the spatial deviation of a set of points from a flight
# path.  It gets the direction of a flight path and provides metrics
# about the deviation of points from that flight path.  It reads the points
# associated with a flight path that (usually) have a scan angle of 19
# degrees and -19 degrees (identified by getflightpath_and_edge_scanangle.py).
#
# INPUT REQUIREMENTS:
# This program reads files produced by getflightpath_and_edge_scanangle.py
# that produces a single file for each flightpath for each data set.
# the informaiton produced is all the points for the best scan angle for each
# flightpath -- usually either -19 (left side of plane) or +19 (right side) --
# sorted by ascending gpstime for a single angle . In addition:
# -- The endpoints of the flightpath will have been trimmed to eliminate
#    some noise that was observed at the start and end of flight paths.
################## COUNTRUNS ##############################################
# This function counts the number of positive and negative runs of deviations
# from a flightline.
def countruns(df):
    numruns=1
    currsign=np.sign(df.loc[0,'deviation'])
    for i in range(1,df.shape[0]):
        if np.sign(df.loc[i,'deviation']) != currsign:
            numruns += 1
            currsign=np.sign(df.loc[i,'deviation'])
    return numruns 
#################### GETCLOSEST ##########################################
def getclosest(xgrid,ygrid,xwide,ywide,dfnoout,outcols):
    dftemp=pd.DataFrame(columns=outcols)
    dftemp.loc[0]=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
    dftemp.loc[0,'xgrid']=xgrid
    dftemp.loc[0,'ygrid']=ygrid               
# Calculate distance from point (xwide,ywide) to other points, get the
# index of the row with the minimum distance.
    closdist=np.array([])
    closdist=((((dfnoout['x'] - xwide) ** 2) + (dfnoout['y'] - ywide) ** 2) ** .5).values
    closeindex=closdist.argmin()
#    print(xgrid,ygrid,xwide,ywide,closeindex,closdist.min(),closdist[0:2])
    dftemp.loc[0,'OldIndex']=closeindex
    dftemp.loc[0,'xactual']=dfnoout.loc[closeindex,'x']
    dftemp.loc[0,'yactual']=dfnoout.loc[closeindex,'y']
#################################################################
# BIGGER FILES USED A COMPACT VERSION THAT DID NOT INCLUDE Z
    dftemp.loc[0,'z']=dfnoout.loc[closeindex,'z']
#################################################################
    dftemp.loc[0,'scan_angle']=dfnoout.loc[closeindex,'scan_angle']
    dftemp.loc[0,'gpstime']=dfnoout.loc[closeindex,'gpstime']
    dftemp.loc[0,'flghtpth']=dfnoout.loc[closeindex,'flghtpth']   
# Get index of minimum value
    return dftemp
#################### MAHALANOBISDIST ####################################
# This function calculates the Mahalanobis distance of a set of points
# using x and y coordinates.  It is used to get rid of spatial otuliers. 
def MahalanobisDist(x, y):
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
# Centre on the mean
    xdiff = np.array([x_i - xy_mean[0] for x_i in x])
    ydiff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([xdiff, ydiff])
# Calculate individual Mahalanobis distances.   
    mahal = []
    for i in range(len(diff_xy)):
        mahal.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return mahal
######################## NUMBPOINTS ################################
# numbpoints determines the number of points that will be used for
# endpoint location averaging.
def numbpoints(df):
    if df.shape[0] >= 5:
        points=5
    else:
        points=df.shape[0]
    return points
######################## PREPARRAYS ################################
# preparrays
########################  MAIN PART OF PROGRAM  ####################
# Import libraries and set up lists for looping.
import pandas as pd
import numpy as np
import os
import math
####################################################################
inpath='C:/LAS_Kim/LAS_Data/LAS_GetEdge/LAS_flghtpths_scanangle/'
outpath='C:/LAS_Kim/LAS_Data/LAS_GetEdge/LASEdgeData/'
summryoutpath='C:/LAS_Kim/LAS_Data/LAS_GetEdge/'
# outfileprefix is for the files for each flightpath
outfileprefix='edgpnts_'
# The following is the name of the summary file.
edgeoutfile='edgepoints_summary.csv'
# The following is for flight path 1 on 2707500n
#infile='2707500n_scanangle_eq_20.csv'
#outfile='v2testedge075_flghtpath1_2707500n.csv'
# The following is for flight path 2 on 2707500n
#infile='2707500n_scanangle_eq_minus20.csv'
#outfile='testedgev075_flghtpath2_2707500n.csv'
# Flightpath 4 2708000n
#infile='2708000n_flghtpth4_scnangl_mnus20.csv'
#infile='2708000n_flghtpth4_scnangl_20.csv'
#infile='2708000n_flghtpth5_plus20.csv'
#outfile='testedgev075_flghtpath5_2708000n_plus20.csv'
#infile='2708000n_flghtpth78_scnangl_mnus20.csv'
#outfile='testedgev075_flghtpath78_2708000n_mnus20.csv'
######################################################################
#################### hyperparameters #################################
# The following hyperparameters control various things.
# -- lineoffset is the number of meters to shift the flightpath to calculate
#    nearest points.
# -- pctoutliers is the % of points that will be removed -- i.e., the n pct
#    highest Mahalanobis distances.
# -- densdist is the distance in m for which the point density around
#    which the local density of a point will be calculated.
# -- elimz is the z value for the percentage of points to eliminate -- i.e.,
#    1.96 will eliminate 2.5% (one tail), 1.645 will eliminate 5%, 1.28155
#    will eliminate 10%, etc.
# -- timegrab is the length of time considered for points to be neighbours
#    of a given point. A larger timegrab will slow processing; a smaller
#    timegrab risks to undercount the true number of neighbours -- especially
#    if densdist increases.
# -- endgrab is the width in m of the band of points that will be used to
#    determine the equation of the line
lineoffset=20
pctoutliers=0.005
densdist=0.5
elimz=0.75
timegrab=0.5
endgrab=1.5
###########################################################################
# Get names of files in the directory and loop through them.
edge_files = os.listdir(inpath)
for k,file in enumerate(edge_files):
# Read the file, sort it by gpstime (even though it should already be sorted),
# convert to numpy array with .values, and calculate Mahalanobis distances
# on x and y so we can eliminate outliers.
    print('\n\n***** Working on file',file,'*****\n     -- File',k+1,'of',len(edge_files))
    dfin=pd.read_csv(inpath+file)
    dfin=dfin.sort_values('gpstime')
    dfin=dfin.reset_index(drop=True)
    xarray=dfin['x'].values
    yarray=dfin['y'].values
    mahal=MahalanobisDist(xarray,yarray)
    dfin['Mahal']=mahal
# Eliminate all observations with the pctoutlier highest Mahalanobis Distance.
    print('\n Start Mahalanobis Screening for Extreme Outliers')
    dfnoout=dfin.sort_values('Mahal')
    dfnoout=dfnoout.reset_index(drop=True)
    keepuptoindex=dfnoout.shape[0]-int(pctoutliers*dfnoout.shape[0])
    dfnoout=dfnoout[0:keepuptoindex]
    dfnoout=dfnoout.sort_values('gpstime')
    dfnoout=dfnoout.reset_index(drop=True)
    nomahaltotal=dfnoout.shape[0]
    print('   Mahalanobis distance:',dfin.shape[0]-nomahaltotal,'outliers discarded')
# Now screen outliers using point density. For each point, count the number
# of points within densdist.  Then eliminate those that are more than -3 
# standard deviations away from the mean number of points. (We only care
# about a sparse density.)
###############################################
    print('\n Start Density Screening')
    for i in range(dfnoout.shape[0]):
# To speed calculations, only calculate distances for points within 0.5
# seconds of the subject.
        dfnoouttemp=dfnoout[(dfnoout['gpstime']>(dfnoout.loc[i,'gpstime']-timegrab/2)) &
                       (dfnoout['gpstime']<(dfnoout.loc[i,'gpstime']+timegrab/2))]
        neighsdist=np.array([])
        xpoint=dfnoouttemp.loc[i,'x']
        ypoint=dfnoouttemp.loc[i,'y']
        closdist=np.array([])
        closdist=((((dfnoouttemp['x'] - xpoint) ** 2) + 
                   (dfnoouttemp['y'] - ypoint) ** 2) ** .5).values
# Get index of first distance > 0.5; this is the number of this
# point's neighbours.
        closdist=np.sort(closdist)
        dfnoout.loc[i,'neighpoints']=np.argmax(closdist>densdist)
        if i%5000==0:
            print('   Working on point',i,'of',dfnoout.shape[0])
#    if i>5000:
#        break
    meanneighs=dfnoout['neighpoints'].mean()
    stdneighs=dfnoout['neighpoints'].std()
# The addition of 0.5 effectively rounds to the next higher integer.
    thresh=meanneighs-elimz*stdneighs+0.5
    dfnoout = dfnoout[dfnoout['neighpoints']>thresh]
    dfnoout=dfnoout.reset_index(drop=True)
    print('Density screening:',nomahaltotal-dfnoout.shape[0],'outliers discarded\n',
          '  Points having',int(thresh),'or fewer neighbours were dropped\n   Mean:',
          round(meanneighs,1),'  Std Dev:',round(stdneighs,1))
#####################################################
# Get max and min x and y to define corners.
    maxx=dfnoout['x'].max()
    maxy=dfnoout['y'].max()
    minx=dfnoout['x'].min()
    miny=dfnoout['y'].min()
# Determine if this line is a "horizontal" (E-W) direction or 
# "vertical" (N-S). The longest axis indicates direction.
    if abs(maxx-minx) < abs(maxy-miny):
# flightpath is north-south
        if dfnoout.loc[0,'y']>dfnoout.loc[dfnoout.shape[0]-1,'y']:
            pathdirect='N2S'
        else:
            pathdirect='S2N'
    else:
        if dfnoout.loc[0,'x']<dfnoout.loc[dfnoout.shape[0]-1,'x']:
            pathdirect='W2E'
        else:
            pathdirect='E2W'
# Get all points within 1 m of the extreme borders in the appropriate 
# end (north and south for a N-S flightpath, e-w for an e-w flightpath)
# WARNING: Using less than 1 m may mean endpoint coordinate is based
# on one point only.
    if 'N' in pathdirect:
# Recall: x coordinates increase from West to East
        dfend1=dfnoout[dfnoout['y']<(miny+endgrab)]
        dfend2=dfnoout[dfnoout['y']>(maxy-endgrab)]
    else:
# Recall: y coordinates increase from South to North
        dfend1=dfnoout[dfnoout['x']>(maxx-endgrab)]
        dfend2=dfnoout[dfnoout['x']<(minx+endgrab)]
# Get endpoints. NOTE: scan_angle should be the same for all rows.
    if 'N' in pathdirect:
# This is a north-south flightpath
        dfend1=dfend1.sort_values('x')
        dfend2=dfend2.sort_values('x')
        dfend1=dfend1.reset_index(drop=True)
        dfend2=dfend2.reset_index(drop=True)
# Determine number of points to use in "location averaging". Default is 5
# unless fewer than 5 points are available.
        onepts=numbpoints(dfend1)
        twopts=numbpoints(dfend2)
        if (pathdirect == 'S2N' and dfend1.loc[0,'scan_angle']>0) or \
            (pathdirect == 'N2S' and dfend1.loc[0,'scan_angle']<0):
# We want points on right side of plane
            dfone=dfend1[['x','y']].tail(onepts)
            dftwo=dfend2[['x','y']].tail(twopts)
        else:
# Get points on left side
            dfone=dfend1[['x','y']].head(onepts)
            dftwo=dfend2[['x','y']].head(twopts)
# E2W and W2E flightpaths
    else:
        dfend1=dfend1.sort_values('y')
        dfend2=dfend2.sort_values('y')
        dfend1=dfend1.reset_index(drop=True)
        dfend2=dfend2.reset_index(drop=True)
# Determine number of points to use in "location averaging". Default is 5
# unless fewer than 5 points are available.
        onepts=numbpoints(dfend1)
        twopts=numbpoints(dfend2)
# Get points on south side of plane
        if (pathdirect == 'W2E' and dfend1.loc[0,'scan_angle']>0) or \
            (pathdirect == 'E2W' and dfend1.loc[0,'scan_angle']<0):
            dfone=dfend1[['x','y']].head(onepts)
            dftwo=dfend2[['x','y']].head(twopts)
# Get points on north side of plane
        else:
            dfone=dfend1[['x','y']].tail(onepts)
            dftwo=dfend2[['x','y']].tail(twopts)
# Get "average endpoints" and equation of the line. In this case flight line
# direction does not matter. If the flightpath is e-w, x are the y coordinates
# and vice versa.
    startx=dfone['x'].mean()
    starty=dfone['y'].mean()
    endx=dftwo['x'].mean()
    endy=dftwo['y'].mean()
    slope=(starty-endy)/abs(startx-endx)
    intercept=(starty+endy)/2-slope*(startx+endx)/2
# Now get the intercept for a line that is so far to the left or right of the 
# direction the lidar looks that all points must be to the left or right.
    if (pathdirect == 'S2N' and dfend1.loc[0,'scan_angle']>0) or \
            (pathdirect == 'N2S' and dfend1.loc[0,'scan_angle']<0):
        paraincpt=(starty+endy)/2-slope*(startx+lineoffset+endx+lineoffset)/2
# Get points on west side of plane
    elif (pathdirect == 'S2N' and dfend1.loc[0,'scan_angle']<0) or \
            (pathdirect == 'N2S' and dfend1.loc[0,'scan_angle']>0):
        paraincpt=(starty+endy)/2-slope*(startx-lineoffset+endx-lineoffset)/2
    # Get points on south side of plane
    elif (pathdirect == 'W2E' and dfend1.loc[0,'scan_angle']>0) or \
            (pathdirect == 'E2W' and dfend1.loc[0,'scan_angle']<0):
#    paraincpt=(starty+endy)/2-slope*(startx-lineoffset+endx-lineoffset)/2
                paraincpt=intercept-lineoffset
# Get points on north side of plane
    else:
#    paraincpt=(starty+endy)/2-slope*(startx-lineoffset+endx-lineoffset)/2
        paraincpt=intercept+lineoffset
# Now go along the range of x variables looking for the values closest to the
# in the direction we want. do this as 1/500th of the range of x. given
# that the tiles are 500 m long, but some flight lines are much shorter
# This will cover many flight lines at 1 m intervals, but could
# give an excessive number of points for some flight paths.
    outcols=['OldIndex','xgrid','ygrid','xactual','yactual','z','scan_angle',
             'gpstime','flghtpth']
    dfout=pd.DataFrame(columns=outcols)
    divisions=500
    for i in range(divisions):
        if (pathdirect == 'S2N' and dfend1.loc[0,'scan_angle']>0) or \
            (pathdirect == 'N2S' and dfend1.loc[0,'scan_angle']<0):
            xgrid=startx-i*(startx-endx)/500
            xwide=xgrid+lineoffset
        else:
            xgrid=startx-i*(startx-endx)/500
            xwide=xgrid-lineoffset
        ywide=slope*xwide+paraincpt
        ygrid=slope*xgrid+intercept
# Find closest point to the wide point.
        dfclosest=getclosest(xgrid,ygrid,xwide,ywide,dfnoout,outcols)
        dfout=pd.concat([dfout,dfclosest],axis=0)
# Return to original order, calculate Euclidian distance from the line,
# sort to keep closest grid point to a duplicate point, drop duplicate points,
# and re-index. 
    dfout=dfout[outcols]
# Calculate the distance between the gridpoints and the actual point, sort
# on this distance and keep the first of each point which should be the
# closest. 
    dfout['deviation']=(((dfout['xgrid']-dfout['xactual'])**2 + 
         (dfout['ygrid']-dfout['yactual'])**2)**0.5)
    dfout=dfout.sort_values('deviation')
    dfout=dfout.drop_duplicates(subset='OldIndex',keep='first')
    dfout=dfout.sort_values('OldIndex')
# Calculate distance of point from flight path.
    dfout['deviation']=(slope*dfout['xactual']-dfout['yactual']+intercept) / \
                        ((slope**2+1)**0.5)
    dfout=dfout.reset_index(drop=True)
# Calculate Euclidian distance from the line.
    dfout.to_csv(path_or_buf=outpath+outfileprefix+file,index_label='Index')
###############################################################
#    if k >= 1:
#        break
###############################################################
# There is now a directory that has individual files for each flight path
# describing the flight lines and deviations from them. Summarise over
# all flight paths.
# Get names of files in the directory and loop through them.
print('\n')
edge_files = os.listdir(outpath)
outcols=['flghtpth','starttime','endtime','startx','starty','endx','endy',
         'pthdist','pthtime',
         'pthazim','speed_kph','ndevpts','maxabsdev','meandev','stddevdev',
         'stderrdev','zmeandev','posdevs','negdevs','numruns','znumruns']
dfout=pd.DataFrame(columns=outcols)
for i,edgename in enumerate(edge_files):
    print('Summarising',edgename)
    dfedgefile=pd.read_csv(outpath+edgename)
    starttime=dfedgefile.loc[0,'gpstime']
    endtime=dfedgefile.loc[dfedgefile.shape[0]-1,'gpstime']
    startx=dfedgefile.loc[0,'xgrid']
    starty=dfedgefile.loc[0,'ygrid']
    endx=dfedgefile.loc[dfedgefile.shape[0]-1,'xgrid']
    endy=dfedgefile.loc[dfedgefile.shape[0]-1,'ygrid']
    pthdist=((startx-endx)**2+(starty-endy)**2)**0.5
    pthtime=endtime-starttime
    pthazim=math.degrees(math.atan2((endx-startx),(endy-starty)))
    if pthazim<0:
        pthazim=360+pthazim
    speed_kph=pthdist/pthtime*3600/1000
    ndevpts=dfedgefile.shape[0]
    maxabsdev=dfedgefile['deviation'].abs().max()
    meandev=dfedgefile['deviation'].mean()
    stddevdev=dfedgefile['deviation'].std()
    stderrdev=stddevdev/(ndevpts**0.5)
    zmeandev=meandev/stderrdev
    posdevs=dfedgefile['deviation'].ge(0).sum()
    negdevs=dfedgefile['deviation'].lt(0).sum()
    numruns=countruns(dfedgefile)
    expnumruns=numruns - ((2*posdevs*negdevs) / (posdevs+negdevs) + 1)
    varexpnumruns = np.sqrt(((2*posdevs*negdevs)*(2*posdevs*negdevs-posdevs-negdevs)) /
                            ((posdevs+negdevs)**2 * (posdevs+negdevs-1)))
    znumruns=expnumruns/varexpnumruns
    dftemp=pd.DataFrame([np.array([edgename,round(starttime,6),round(endtime,6),
                round(startx,2),round(starty,2),round(endx,2),round(endy,2),
                round(pthdist,1),round(pthtime,6),round(pthazim,1),
                round(speed_kph,1),ndevpts,round(maxabsdev,2),round(meandev,2),
                round(stddevdev,2),round(stderrdev,2),round(zmeandev,3),
                posdevs,negdevs,numruns,round(znumruns,4)])],columns=outcols)
#    if i >= 0:
#        break
    dfout=pd.concat([dfout,dftemp],axis=0)
dfout.to_csv(summryoutpath+edgeoutfile,index=False)