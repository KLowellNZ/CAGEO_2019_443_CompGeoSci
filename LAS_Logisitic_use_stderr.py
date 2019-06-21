# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 13:26:12 2018

@author: klowell
"""

#%%
#############################################################################
# This program reads a LAS csv file that has been prepared for model
# fitting using logisitc regression. It fits a logistic model and outputs
# various summary statistics.  Validation is done by k-fold validation.
# Though this is time-consuming, it selects the optimal penalty. (The
# alternative is splitting the data into 80/20 train/test which requires
# trying multiple penalities in hopes of choosing the right one.)
#
# Originally I tried using reportlab.pdfgen to output everything --
# graphs and tables -- to a pdf but could not get it to work. Library
# loadings remain in this code, but the (failed) code to write has
# been deleted.
#################### ACCURACY#####################################
# This function gets the ACCURACY statistics given two dfs of
# predicted and actual.
def ACCURACY(pred,actual):
    acc=accuracy_score(pred,actual)
    prec=precision_score(pred,actual,average='weighted')
    rec=recall_score(pred,actual,average='weighted')
    return acc, prec,rec
############# ADD_AZIM_SLOPE_ORTHOG ####################################
# This function calculates the orthogonality of a pulse
# to the slope and azimuth of the seafloor. A dataframe with the added
# attributes is returned.
def add_azim_slope_orthog(df):
# Seafloor aspect deviation. A pulse is orthogonal if its azimuth is
# directly opposite (180 deg) from the seafloor aspect. Scale to between 
# 0 and 100.
############################# 1M GRID ###########################
    df['azimdiff_1m']=np.abs(df['azim2plse']-df['aspect_1m'])
    df['asp_orthog_1m']=np.cos(np.radians(df['azimdiff_1m']))*-1
    df['asp_orthog_1m']=((df['asp_orthog_1m']+1)/2)*100
# Slope steepness deviation. (The cosine is squared to accentuate the deviation.)
# This will be scaled between 0 and 1 in the main program.
    df['slp_degrees_1m']=np.degrees(np.arctan(df['slopepct_1m']/100))
# Check if the hill is facing towards or away from us.
    df['dir_flag_1m']=np.where((df['azimdiff_1m']<90) | (df['azimdiff_1m']>270),-1,1)
    df['slp_orthog_1m']=np.abs(df['inciangle']-(df['dir_flag_1m']*df['slp_degrees_1m']))
    df['slp_orthog_1m']=np.cos(np.radians(df['slp_orthog_1m']))**2
############################# 5M GRID ###########################
    df['azimdiff_5m']=np.abs(df['azim2plse']-df['aspect_5m'])
    df['asp_orthog_5m']=np.cos(np.radians(df['azimdiff_5m']))*-1
    df['asp_orthog_5m']=((df['asp_orthog_5m']+1)/2)*100
# Slope steepness deviation. (The cosine is squared to accentuate the deviation.)
# This will be scaled between 0 and 1 in the main program.
    df['slp_degrees_5m']=np.degrees(np.arctan(df['slopepct_5m']/100))
# Check if the hill is facing towards or away from us.
    df['dir_flag_5m']=np.where((df['azimdiff_5m']<90) | (df['azimdiff_5m']>270),-1,1)
    df['slp_orthog_5m']=np.abs(df['inciangle']-(df['dir_flag_5m']*df['slp_degrees_5m']))
    df['slp_orthog_5m']=np.cos(np.radians(df['slp_orthog_5m']))**2
# Get rid of intermediate variables.
    df=df.drop(['slp_degrees_1m','dir_flag_1m','slp_degrees_5m','dir_flag_5m'],axis=1)
    return df
################# CROSSTABSTATS #################################
# This function calculates total accuracy and user's and producer's
# accuracy for each class of a crosstab confusion matrix.
def crosstabstats(NCASclasses,crosstab,confmatrixfile):
# Get total observations, then sum of columns, rows and diagonal.
    total=sum(crosstab.sum(0))
    sumcols=crosstab.sum(0)
    sumrows=crosstab.sum(1)
    sumdiag=0
    for i in range(len(NCASclasses)):
        try:
            sumdiag += crosstab.iloc[i,i]
        except:
            continue
# Now get and print stats.
    totacc=round(sumdiag/total*100,1)
    print('\n Total accuracy %:',totacc)
    print('\n Total accuracy %:',totacc,file=confmatrixfile)
    useracc=[]
    prodacc=[]
    print("\n Accuracy: User's  Producer's")
    print("\n Accuracy: User's  Producer's",file=confmatrixfile)
    for i in range(len(sumcols)):
# If a row sums to zero, the model predicted no pixels/polygons for that
# class.
        try:
            useracc.append(round((crosstab.iloc[i,i]/sumrows[i]*100),1))
        except:
            useracc.append(-99.9)
# A column sum of zero indicates there were none of this class in the data base.
# This should be impossible. However, if there is no diagonal element for the
# class, the model did not predict any meaning the producer's accuracy for
# the class is zero.
        try:
            prodacc.append(round((crosstab.iloc[i,i]/sumcols[i]*100),1))
        except:
            prodacc.append(0.0)
        print(' ',NCASclasses[i].ljust(10),useracc[i],'    ',prodacc[i])
        print(' ',NCASclasses[i].ljust(10),useracc[i],'    ',prodacc[i],
                  file=confmatrixfile)
    return totacc,useracc,prodacc
##################### LOGIT_PVALUES ##################################
# Get standard errors of coefficients.
def logit_pvalues(model, x):
    Xtemp=x.values
# For some reason, this can produce negative probabilities. Take abs value.
    p = abs(model.predict_proba(Xtemp))
    n = len(p)
# Copy all model coefficients (except Intercept) to a list. Store
# names of varialbes for each coefficient.
    coefflist=model.coef_[0]
    varlist=x.columns
# Now eliminate zero coefficients, their names, and their columns.
    for i in range(len(coefflist)-1,-1,-1):
        if coefflist[i]==0:
            coefflist=np.delete(coefflist,i)
            varlist=np.delete(varlist,i)
            x.drop(x.columns[i],axis=1,inplace=True)
# The list of coefficient values are stored in a list.
    coefflist=[coefflist]
    Xmatrix=x.values
    m = len(coefflist[0]) + 1
# coefs is a list of all non-zero coefficient values. Only use if z-scores
# are calculated in this function.
#    coefs = np.concatenate([model.intercept_, coefflist[0]])
    x_full = np.matrix(np.insert(np.array(Xmatrix), 0, 1, axis = 1))
    ans = np.zeros((m, m))
# Get co-variance matrix, invert it, and take the square root of the
# diagonal elements. This is the coefficient standard errors.
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
# If the matrix cannot be inverted because it is singular, use the following
# statement (approximate inversion) instead of the previous one.
#    vcov = np.linalg.pinv((np.matrix(ans)))
    SE_list = np.sqrt(np.diag(vcov))
    SE_list[np.isnan(SE_list)]=-99.99
    varlist=np.insert(varlist,0,'Intercept')
    SEdict = dict(zip(varlist, SE_list))
    return SEdict
####################### PLOT_INVROC #####################################
# Plot_invroc plots a true psotive vs. Specificity (1-false positive) 
# curve that indicates where the tpr and fpr are "in balance" -- i.e., 
# mutually minimised.
def plot_invroc(fpr,tpr,thr,ax):
# Optimal threshold is where tpr is high and fpr is low -- i.e., where
# tpr - (1-fpr) is closest to zero.
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i),
                '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr),
                index = i), 'thresholds' : pd.Series(thr, index = i)})
    opt_thresh=roc.iloc[(roc.tf-0).abs().argsort()[:1]].reset_index(drop=True)
# Plot tpr vs 1-fpr
    ax.plot(roc['thresholds'],roc['tpr'],color='blue')
    ax.plot(roc['thresholds'].abs(), roc['1-fpr'],color = 'red')
    ax.set_xlabel('Threshold (p > threshold is classified Bathymetry)')
    ax.set_ylabel('Classificaiton Rate')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_title('Optimum Threshold Value (Intersection of TPR(Blue) and TNR(Red))')
    ax.set_xticks(np.arange(0,1.1,step=0.1))
    ax.set_yticks(np.arange(0,1.1,step=0.1))
    return opt_thresh, ax
####################### PLOT_ROC #####################################
# plot_roc plots an roc curve and a diagonal
def plot_roc(actual, predicted, filename,ax):
    fpr, tpr, thr = roc_curve(actual, predicted)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc,color='blue')
    ax.plot([0, 1], [0, 1], 'k--',color='gray')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks(np.arange(0,1.1,step=0.1))
    ax.set_yticks(np.arange(0,1.1,step=0.1))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC (Bathymetry)'+filename)
    ax.legend(loc='lower right')
# Return fpr, tpr, thr so they do not have to be re-calculated.
    return fpr, tpr, thr, ax
#################### PLSE_FRM_HEADING ##############################
# This calculates the difference of a pulse relative to the plane heading.
# Values are between 0 and 90 because we only want the angle -- whether
# it is from the front or back of the plane is in scan_direct.
def plse_frm_heading(df):
    df['absdiffrads']=np.abs(np.radians(df['heading'])-np.radians(df['azim2plse']))
    df['pls_frm_hdng']=np.where(df['absdiffrads']<(2*np.pi-df['absdiffrads']),
          df['absdiffrads'],(2*np.pi-df['absdiffrads']))
    df['pls_frm_hdng']=np.degrees(df['pls_frm_hdng'])
    df['pls_frm_hdng']=np.where(df['pls_frm_hdng']<=90,df['pls_frm_hdng'],
          180-df['pls_frm_hdng'])
    df.drop('absdiffrads',axis=1,inplace=True)
    return df
#################### PRINTCROSSTAB ###################################
# This function prints the cross-tabulation of "truth" vs predicted.
def printcrosstab(label,NCASclasses,crosstab,confmatrixfile):

    print(label.ljust(35),'\n          ',end=' ')
    print(label.ljust(35),'\n          ',end=' ',file=confmatrixfile)
# Get matrix total and row and column totals.
    total=sum(crosstab.sum(0))
    sumcols=crosstab.sum(0)
    sumrows=crosstab.sum(1)
# Print column labels
    for ii in range(len(NCASclasses)):
#       stringelem=NCASclasses[ii].astype(str)
        stringelem=NCASclasses[ii]
        print(stringelem.rjust(9),end=' ')
        print(stringelem.rjust(9),end=' ',file=confmatrixfile)
# Print column label for total.
    print('    Total',end=' ')
    print('    Total',end=' ', file=confmatrixfile)
# Print the row label and then the elements of the row.
    for ii in range(len(NCASclasses)):
        stringelem=NCASclasses[ii]
        print('\n',stringelem.ljust(9),end=' ')
        print('\n',stringelem.ljust(9),end=' ',file=confmatrixfile)
        for jj in range(len(NCASclasses)):
            try:
                stringelem=crosstab.iloc[ii,jj].astype(str)
            except:
                stringelem='0'
            print(stringelem.rjust(9),end=' ')
            print(stringelem.rjust(9),end=' ',file=confmatrixfile)
# Print row totals.
        print(sumrows[ii].astype(str).rjust(9),end=' ')
        print(sumrows[ii].astype(str).rjust(9),end=' ',file=confmatrixfile)
# Print row label for total and total for columns.  (YES columns.)
# NOTE: Total had to be converted to string using str(total) because it is
# type int rather than type int64 (which takes int64var.astype(str).
    print('\n   Total  ',sumcols[0].astype(str).rjust(9),sumcols[1].astype(str).rjust(9),
          str(total).rjust(9), end=' ')
    print('\n   Total  ',sumcols[0].astype(str).rjust(9),sumcols[1].astype(str).rjust(9),
          str(total).rjust(9), end=' ', file=confmatrixfile)
########################  MAIN PART OF PROGRAM  ####################
# Import libraries.
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st
import time
############ TEXT PDF LIBARARIES #######################################
#from reportlab.pdfgen import canvas
#from reportlab.lib.pagesizes import letter, landscape
###################### FIGURE PDF LIBARARIES ###########################
#import io
#from reportlab.lib.utils import ImageReader
#######################################################################
################## PLOT/FIGURE LIBRARIES ###############################
#from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
####################################################################
in_out_path='C:/LAS_Kim/LAS_Data/LAS_for_Analysis/'
#file_list=['df2016_430000e_2707500n_all_for_final_analysis.csv']
#file_list= ['df2016_430000e_2707500n_w_SBET_edge_scan_drctn_xy.csv']
#file_list= ['df2016_430000e_2707500n_w_SBET_edge_scan_drctn_xy.csv',
#            'df2016_426000e_2708000n_w_SBET_edge_scan_drctn_xy.csv']
file_list= ['df2016_430000e_2707500n_all_for_final_analysis.csv',
            'df2016_420500e_2728500n_all_for_final_analysis.csv',
            'df2016_428000e_2719500n_all_for_final_analysis.csv',
            'df2016_426000e_2708000n_all_for_final_analysis.csv']
#Classdict to re-label rows. Assumes a binary -- rather than multi-
# nomial classification.
classdict={0:'NotBathy',1:'Bathymt'}
#################### hyperparameters #################################
# .txt file will contain tabular information and the .pdf will 
# have the ROC curve.
############################################################################
varlabel1 = 'Variables: Reduced main effects plus 5m elevation -- no topo nor orthogs'
varlabel2=' THIS IS THE BASE MODEL PLUS 5M ELEV'
#varlabel2 = '     but not slope nor aspect.'
############################################################################
allnameprefix='l_FinalAnalysis1_RLR_IndividTiles_ReduMainEff_5m_elev'
alltilesfilename=in_out_path+allnameprefix+'.txt'
alltilesfile= open(alltilesfilename,'w')
alltilespdfname=in_out_path+allnameprefix+'.pdf'
alloutpdf=PdfPages(alltilespdfname)
###########################################################################
textfilenames=[]
# Loop through the files.
for k,file in enumerate(file_list):
#################################################################
#    if k==1:
#       break
#################################################################
    time_tic=time.time()
# Read the file and set up output pdf file.
    print('\n\n***** Reading file',file,'*****')
######################################################################
################## USE A SMALL FILE FOR DEBUGGING ####################
    dfin=pd.read_csv(in_out_path+file)
#    dfin=pd.read_csv(in_out_path+file,nrows=100000)
###########################################################################
######## CHANGE HERE AND BELOW IF NOT NORMALISING VARIABLES ###############
    tilenameprefix='l_FinalAnalysis1_RLR_IndividTiles_ReduMainEff_5m_elev_'
# dfbathnameprefix is only used if I am outputting a slice of the confusion
# matrix for subsequent analysis.
#    dfbathnameprefix='FinalAnalysis1_RLR_AllMainEffects_1m_5m_grids_'
    confmatrixname= in_out_path+tilenameprefix+file.replace('.csv','.txt')
    tilepdfname= in_out_path+tilenameprefix+file.replace('.csv','.pdf')
    tileoutpdf=PdfPages(tilepdfname)
    textfilenames.append(confmatrixname)
#    confmatrixname= in_out_path+'Test_ConfMtrx_log_nrmlsd_'+file.replace('.csv','_l2.txt')
###########################################################################
    # Read file and set up output confusion matrix file.
    confmatrixfile= open(confmatrixname,'w')
# Create the required variables.
    dfin['abs_devia']=dfin['deviation'].abs()
######################################################################
########## FOR DEBUGGING MAKE BATHYMETRY=1 FOR SELECTED ROWS##########
    dfin['Bathymetry']=np.where(dfin['class']=='Bth', 1, 0)
#    for k in range(250,1950):
#        dfin.loc[k,'Bathymetry']=1
#################### SHUFFLE ###############
# Shuffle to mitigate impacts of imbalanced classes.
    dfin=shuffle(dfin)
############################################
############ ENGINEER FEATURES ###########################################
# Lack of aspect and slope orthogonality measures asp_orthog and slp_orthog.
    dfin=add_azim_slope_orthog(dfin)
# Topo interactions
    dfin['slp_asp_orthog_1m']=dfin['slp_orthog_1m']*dfin['asp_orthog_1m']
    dfin['slp_asp_orthog_5m']=dfin['slp_orthog_5m']*dfin['asp_orthog_5m']
# Pulse angle relative to plane heading.  Add pls_frm_hdng.
    dfin=plse_frm_heading(dfin)
################### CREATE INTERACTIONS ##################################
#    dfin['absSA_tms_absDevia']=dfin['abs_scnangl']*dfin['abs_devia']
#    dfin['absSA_tms_Rtrn_no']=dfin['abs_scnangl']*dfin['return_no']
#    dfin['absDevia_tms_Rtrn_no']=dfin['abs_devia']*dfin['return_no']
#    dfin['absSA_tms_nmrtrns']=dfin['abs_scnangl']*dfin['num_returns']
#    dfin['absSA_tms_stdYPR']=dfin['abs_scnangl']*dfin['stdYwPtRl']
#    dfin['absSA_tms_last']=dfin['abs_scnangl']*dfin['last']
#    dfin['nmrtrns_tms_stdYPR']=dfin['num_returns']*dfin['stdYwPtRl']
#    dfin['absSA_tms_scndrctn']=dfin['abs_scnangl']*dfin['scan_direct']
#    dfin['scndrctn_tms_Rtrn_no']=dfin['scan_direct']*dfin['return_no']
# NOTE: No (log) transforms are employed because it was determined that
# these do not improve classification.
#
# Create an x data frame having the independent variables and then a y
# data frame having the binary dependent variable. Normalise the
# x variables between 0 and 100 and convert both to a matrix.
######################################################################
###### (SEMI-)FULL VARIABLE LIST: All main effects and selected
# interactions.
#    Xcols=['num_returns','return_no','single','first_of_many','last',
#           'rela_return_num','stdXYZ','stdYwPtRl','maxabsdev','stddevdev',
#           'znumruns','abs_devia','abs_scnangl','absSA_tms_absDevia',
#           'absSA_tms_Rtrn_no','absDevia_tms_Rtrn_no','absSA_tms_nmrtrns',
#           'absSA_tms_stdYPR','absSA_tms_last','nmrtrns_tms_stdYPR',
#           'last_of_many','scan_direct','absSA_tms_scndrctn',
#           'scndrctn_tms_Rtrn_no']
#################### MAIN EFFECTS WITH CAVEAT ############################
# CAVEAT: maxabsdev, stddevdev, znumruns are the same for individual
# flightpaths and therefore identical when standardised. If more than one
# is retained in the equation, the covariance matrix is singular,it cannot
# be inverted, and coeff standard errors cannot be calculated. Therefore
# only include one of the three variables -- arbitrarily keep maxabsdev
# and eliminate stddevdev and znumruns. (If this does not work, in the 
# function logit_pvalues there is an alternative statement to do an 
# approximate inversion.)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia','scan_direct',
#           'azim2plse','inciangle','pls_frm_hdng',
#           'elevintrp_1m','slopepct_1m','aspect_1m','asp_orthog_1m','slp_orthog_1m',
#           'slp_asp_orthog_1m',
#           'elevintrp_5m','slopepct_5m','aspect_5m','asp_orthog_5m','slp_orthog_5m',
#           'slp_asp_orthog_5m']
############# MAIN EFFECTS ONLY NO TOPO, ORTHOGS, OR INTERACTIONS #############
# NOTE:: maxabsdev, stddevdev, znumruns are the same for individual
# flightpaths and therefore identical when standardised. For RLR, if more than one
# is retained in the equation, the covariance matrix is singular,it cannot
# be inverted, and coeff standard errors cannot be calculated. (13 vars)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
#           'scan_direct','azim2plse','inciangle','pls_frm_hdng']
################# REDUCED MAIN EFFECTS NO SLOPE ASPECT MAINS  1M ##############
# The following were used to examine only the effects of slope and aspect
# orthogonality without the impacts of slope and aspect main effects (but
# including elevation). (17 vars)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
#           'scan_direct','azim2plse','inciangle','pls_frm_hdng',
#           'elevintrp_1m','asp_orthog_1m','slp_orthog_1m',
#           'slp_asp_orthog_1m']
############# F. REDUCED LIST OF MAIN EFFECTS ALL TOPO & ORTHOG 5M ############
# The following is for 5m grids -- topo main effects and topo orthog main
# effects. (19 vars)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
#           'scan_direct','azim2plse','inciangle','pls_frm_hdng',
#           'elevintrp_5m','slopepct_5m','aspect_5m',
#            'asp_orthog_5m','slp_orthog_5m','slp_asp_orthog_5m']
########## G. REDUCED LIST OF MAIN EFFECTS ALL TOPO BUT NOT ORTHOG 5M #########
# The following is for 5m grids -- topo main effects and topo orthog main
# effects. (16 vars)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
#           'scan_direct','azim2plse','inciangle','pls_frm_hdng',
#           'elevintrp_5m','slopepct_5m','aspect_5m']
########## H. REDUCED LIST OF MAIN EFFECTS ALL TOPO BUT NOT ORTHOG 5M #########
# The following is for 5m grids -- topo main effects and topo orthog main
# effects. (17 vars)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
#           'scan_direct','azim2plse','inciangle','pls_frm_hdng',
#           'elevintrp_5m','asp_orthog_5m','slp_orthog_5m','slp_asp_orthog_5m']
######### I. REDUCED LIST OF MAIN EFFECTS ALL TOPO BUT NOT ORTHOG 1M&5M #######
# The following is for 5m grids -- topo main effects and topo orthog main
# effects. (19 vars)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
#           'scan_direct','azim2plse','inciangle','pls_frm_hdng',
#           'elevintrp_1m','slopepct_1m','aspect_1m',
#           'elevintrp_5m','slopepct_5m','aspect_5m']
########## J. REDUCED LIST OF MAIN EFFECTS 1&5M ORTHOG BUT NOT TOPO ##########
# The following is for 5m grids -- main effects and orthog main effects but
# not slope and aspect. (21 vars)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
#           'scan_direct','azim2plse','inciangle','pls_frm_hdng',
#           'elevintrp_1m','asp_orthog_1m','slp_orthog_1m','slp_asp_orthog_1m',
#           'elevintrp_5m','asp_orthog_5m','slp_orthog_5m','slp_asp_orthog_5m']
####################### K. MAIN EFFECTS AND 1M ELEVATION #####################
# NOTE:: maxabsdev, stddevdev, znumruns are the same for individual
# flightpaths and therefore identical when standardised. For RLR, if more than one
# is retained in the equation, the covariance matrix is singular,it cannot
# be inverted, and coeff standard errors cannot be calculated. This is
# not an issue with MLP so all three variables are retained here. (14 vars)
# MLP Structure: (20,20,20,20)
#    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
#           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
#           'scan_direct','azim2plse','inciangle','pls_frm_hdng','elevintrp_1m']
####################### L. MAIN EFFECTS AND 5M ELEVATION #####################
# NOTE:: maxabsdev, stddevdev, znumruns are the same for individual
# flightpaths and therefore identical when standardised. For RLR, if more than one
# is retained in the equation, the covariance matrix is singular,it cannot
# be inverted, and coeff standard errors cannot be calculated. This is
# not an issue with MLP so all three variables are retained here. (14 vars)
# MLP Structure: (20,20,20,20)
    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
           'scan_direct','azim2plse','inciangle','pls_frm_hdng','elevintrp_5m']
####################### M. MAIN EFFECTS AND 1&5M ELEVATION #####################
# NOTE:: maxabsdev, stddevdev, znumruns are the same for individual
# flightpaths and therefore identical when standardised. For RLR, if more than one
# is retained in the equation, the covariance matrix is singular,it cannot
# be inverted, and coeff standard errors cannot be calculated. This is
# not an issue with MLP so all three variables are retained here. (14 vars)
    Xcols=['num_returns','return_no','first_of_many','last','last_of_many',
           'stdXYZ','stdYwPtRl','maxabsdev','abs_devia',
           'scan_direct','azim2plse','inciangle','pls_frm_hdng','elevintrp_1m',
           'elevintrp_5m']
#######################################################################
# The following variable list removes maxabsdev, stddevdev, absdevia, 
# znumruns and interactions. This is necessary for testing because these
# have no variability over 1000 rows.
#    Xcols=['num_returns','return_no','single','first_of_many',
#           'last','last_of_many','rela_return_num','stdXYZ','stdYwPtRl',
#           'stdTOT','abs_scnangl','absSA_tms_Rtrn_no']
#    Xcols=['num_returns','last','abs_scnangl']
#    Xcols=['absSA_tms_Rtrn_no','absSA_tms_absDevia','abs_scnangl']
######################################################################
# Create df with variables we want and get their max and min for subsequent
# printing.
    dfX=dfin[Xcols]
    dfXmin=dfX.min()
    dfXmax=dfX.max()
######################### DO NOT NORMALISE FOR TESTING #################
# EVEN IF NOT NORMALISED, THE MATRIX/DF IS CALLED XNORMALMATRIX FOR
# CONSISTENCY BELOW WITH FUNCTION CALLS, ETC.
    dfXnormal=(dfX-dfX.min())/(dfX.max()-dfX.min())*100
#    dfXnormal=(dfX-dfXmin)/(dfXmax-dfXmin)*100
    dfXnormal=dfXnormal.dropna(axis='columns')
    Xcols=dfXnormal.columns
    Xnormalmatrix=dfXnormal.values
#    Xnormalmatrix=dfX.values
########################################################################    
    Ymatrix=dfin[['Bathymetry']].values.flatten()
    print('\tFitting logistic model....')
# Fit model and put chisquared and associated p values.
    lmodel=LogisticRegressionCV(solver='saga',cv=5,penalty='l1').fit(Xnormalmatrix,Ymatrix)
    print('\tModel now fit. Summarising....')
# Get coefficient p values in a dictionary using variable names as the keys.
    SEdict = logit_pvalues(lmodel, dfXnormal)
# Get components for McFadden's pseudo r-squared.
    Yprob=lmodel.predict_proba(Xnormalmatrix)
    model_logloss=log_loss(Ymatrix,Yprob)
# Fit null logistic model.
    Xnull=[]
    for i in range(dfX.shape[0]):
        Xnull.append([1])
    Xnull_matrix=np.asarray(Xnull)
    nullmodel= LogisticRegressionCV(solver='saga',penalty='l1',
                fit_intercept=False).fit(Xnull_matrix,Ymatrix)
    Ynullprob=nullmodel.predict_proba(Xnull_matrix)
    null_logloss=log_loss(Ymatrix,Ynullprob)
    rsqrd=1-model_logloss/null_logloss
# Plot outputs. This is wastefile since values are (re-)calculated later,
# but it is protection from having the program crash at the very end.
# fpr,tpr, and thr are returned from the plotting routines so that they
# do not have to be re-calcualted. fprnull, tprnull, and thrnull serve
# no purpose.
    null_preds=np.ones(len(Ymatrix))
    plt.rc('font',size=6)
    fig,axs=plt.subplots(nrows=1,ncols=2,sharex=True,figsize=(10.5,5))
    fprnull,tprnull,thrnull,axs[0] = plot_roc(Ymatrix,null_preds,file,axs[0])
    fpr,tpr,thr,axs[0] = plot_roc(Ymatrix,Yprob[:,1],file,axs[0])
# Get true positive rate and and false positive rate for threshold = 0.50.
# thr is a list of thresholds starting close to 1.0 and finishing close to 0.0.
# find first thresh < 0.5 and take the row preceding it.
    for kk in range(len(thr)):
        if thr[kk]<0.500:
            break
    modelfpr=fpr[kk-1]
    modeltpr=tpr[kk-1]
# Now plot the "inverse" ROC curve and identify the optimum threshold --
# i.e., where true positive rate and false positive rate are both minimised.
# opt_thresh is a dataframe containing the optimum cutoff point.
    opt_thresh,axs[1]=plot_invroc(fpr,tpr,thr,axs[1])
    tileoutpdf.savefig(fig)
    alloutpdf.savefig(fig)
    fig.clf()
# Make df with coefficient values, standard errors, and variable names sorted by
# abs(coeff values) except the intercept.
#    dfcoeffs=pd.DataFrame({'Coeff':lmodel.intercept_,'Variable':'Intercept',
#                           'stderr':SEdict['Intercept'],'Zscore':0.,'pvalue':1.})
    dfcoeffs=pd.DataFrame({'Coeff':lmodel.intercept_,'Variable':'Intercept',
        'stderr':SEdict['Intercept'],'Zscore':lmodel.intercept_/SEdict['Intercept'],
        'pvalue':(1-st.norm.cdf(abs(lmodel.intercept_/SEdict['Intercept'])))*2})
    dftemp=pd.DataFrame({'Coeff':lmodel.coef_.flatten(),'Variable':Xcols,
                         'stderr':-99.99})
    dftemp['stderr']=dftemp['Variable'].map(SEdict)
    dftemp['Zscore']=dftemp['Coeff']/dftemp['stderr']
    dftemp['pvalue']=(1-st.norm.cdf(abs(dftemp['Zscore'])))*2
    dftemp['absZ']=dftemp['Zscore'].abs()
    dftemp.sort_values(by=['absZ'],inplace=True,ascending=False)
#    dftemp.sort_values(by=['pvalue'],inplace=True,ascending=True)
    dfcoeffs=pd.concat([dfcoeffs,dftemp],axis=0).reset_index(drop=True)
# Map stderrs using SEdict.
    dfcoeffs['stderr'].fillna(-99.9999,inplace=True)
    dfcoeffs['Zscore'].fillna(0.,inplace=True)
    dfcoeffs['pvalue'].fillna(1.,inplace=True)
# Get confusion matrix.
    pred_lmodel=lmodel.predict(Xnormalmatrix)
    modlaccu,modlprec,modlrecl=ACCURACY(pred_lmodel,Ymatrix)
    modelcrosstab=pd.crosstab(pred_lmodel,Ymatrix)
    modelclasses=list(modelcrosstab)
    for k in range(len(modelclasses)):
        modelclasses[k]=classdict[modelclasses[k]]
# Print output
    print('\n\n********** LASSO LOGISTIC_CV MODEL **********\nTile',file,'\tPenalty(l1 - lasso):',
      round(lmodel.C_[0],4),'\n',varlabel1,varlabel2,'\n\tGlobal Accuracy:',
      round(modlaccu,3),
      '\tPrecision:',round(modlprec,3),'\tRecall:',round(modlrecl,3),
      '\tPseudo r^2:',round(rsqrd,3))
    print('\n********** LASSO LOGISTIC_CV MODEL **********\nTile',file,'\tPenalty(l1 - lasso):',
      round(lmodel.C_[0],4),'\n',varlabel1,varlabel2,'\n\tGlobal Accuracy:',
      round(modlaccu,3),
      '\tPrecision:',round(modlprec,3),'\tRecall:',round(modlrecl,3),
      '\tPseudo r^2:',round(rsqrd,3), file=confmatrixfile)
    print('\n********** LASSO LOGISTIC_CV MODEL **********\nTile',file,'\tPenalty(l1 - lasso):',
      round(lmodel.C_[0],4),'\n',varlabel1,varlabel2,'\n\tGlobal Accuracy:',
      round(modlaccu,3),
      '\tPrecision:',round(modlprec,3),'\tRecall:',round(modlrecl,3),
       '\tPseudo r^2:',round(rsqrd,3), file=alltilesfile)
    print('\n\t',len(Xcols),'variables:',Xcols)
    print('\n\t',len(Xcols),'variables:',Xcols, file=confmatrixfile)
    print('\n\t',len(Xcols),'variables:',Xcols, file=alltilesfile)
# Print user's and producer's accuracy for all classes for conventional
# confusion matrix -- i.e., decision threshold =0.50.
    print('\nCONFUSION MATRIX  Threshold = 0.50\n\t(FPR:', round(modelfpr,3),
        '  TPR:',round(modeltpr,3),'  TNR(1-FPR):',round(1-modelfpr,3),
        '  diff(TNR-TPR:',round(abs(1-modelfpr-modeltpr),4),')',end='')
    print('\nCONFUSION MATRIX  Threshold = 0.50\n\t(FPR:', round(modelfpr,3),
        '  TPR:',round(modeltpr,3),'  TNR(1-FPR):',round(1-modelfpr,3),
        '  diff(TNR-TPR:',round(abs(1-modelfpr-modeltpr),4),')',end='',file=confmatrixfile)
    print('\nCONFUSION MATRIX  Threshold = 0.50\n\t(FPR:', round(modelfpr,3),
        '  TPR:',round(modeltpr,3),'  TNR(1-FPR):',round(1-modelfpr,3),
        '  diff(TNR-TPR:',round(abs(1-modelfpr-modeltpr),4),')',end='',file=alltilesfile)
    printcrosstab('\n   PRED    /   ACTUAL',modelclasses,
              modelcrosstab,confmatrixfile)
    printcrosstab('\n   PRED    /   ACTUAL',modelclasses,
              modelcrosstab,alltilesfile)
    globalacc,useracc,prodacc= \
        crosstabstats(modelclasses,modelcrosstab,confmatrixfile)
    globalacc,useracc,prodacc= \
        crosstabstats(modelclasses,modelcrosstab,alltilesfile)
    print('\nOPTIMUM THRESHOLD (all obs w/ p>Optimum is Bathymetry):',
        round(opt_thresh.loc[0,'thresholds'],3),'\n\t(FPR:', round(opt_thresh.loc[0,'fpr'],3),
        '  TPR:',round(opt_thresh.loc[0,'tpr'],3),'  TNR(1-FPR):',round(opt_thresh.loc[0,'1-fpr'],3),
        '  diff(TNR-TPR:',round(abs(1-opt_thresh.loc[0,'fpr']-opt_thresh.loc[0,'tpr']),4),')',)
    print('\nOPTIMUM THRESHOLD (all obs w/ p>Optimum is Bathymetry):',
        round(opt_thresh.loc[0,'thresholds'],3),'\n\t(FPR:',round(opt_thresh.loc[0,'fpr'],3),
        '  TPR:',round(opt_thresh.loc[0,'tpr'],3),'  TNR(1-FPR):',round(opt_thresh.loc[0,'1-fpr'],3),
        '  diff((TNR-TPR:',round(abs(1-opt_thresh.loc[0,'fpr']-opt_thresh.loc[0,'tpr']),4),')',
          file=confmatrixfile)
    print('\nOPTIMUM THRESHOLD (all obs w/ p>Optimum is Bathymetry):',
        round(opt_thresh.loc[0,'thresholds'],3),'\n\t(FPR:',round(opt_thresh.loc[0,'fpr'],3),
        '  TPR:',round(opt_thresh.loc[0,'tpr'],3),'  TNR(1-FPR):',round(opt_thresh.loc[0,'1-fpr'],3),
        '  diff((TNR-TPR:',round(abs(1-opt_thresh.loc[0,'fpr']-opt_thresh.loc[0,'tpr']),4),')',
        file=alltilesfile)
# Now print confusion matrix based on the optimum threshold.
    throptimum=opt_thresh.loc[0,'thresholds']
    pthresh=abs(lmodel.predict_proba(Xnormalmatrix))
    thrlist=[]
    for kl in range(len(pthresh)):
        if pthresh[kl][1]>=throptimum:
            thrlist.append(1)
        else:
            thrlist.append(0)
    Ythrpred=np.array(thrlist)
    threshcrosstab=pd.crosstab(Ythrpred,Ymatrix)
    throptimum=opt_thresh.loc[0,'thresholds']
    print('\nCONFUSION MATRIX  Threshold =',round(throptimum,3),end='')
    print('\nCONFUSION MATRIX  Threshold =',round(throptimum,3),end='',file=confmatrixfile)
    print('\nCONFUSION MATRIX  Threshold =',round(throptimum,3),end='',file=alltilesfile)
    printcrosstab('\n   PRED    /   ACTUAL',modelclasses,
              threshcrosstab,confmatrixfile)
    printcrosstab('\n   PRED    /   ACTUAL',modelclasses,
              threshcrosstab,alltilesfile)
    thrglobalacc,thruseracc,thrprodacc= \
        crosstabstats(modelclasses,threshcrosstab,confmatrixfile)
    thrglobalacc,thruseracc,thrprodacc= \
        crosstabstats(modelclasses,threshcrosstab,alltilesfile)
# Print coefficient values.
    print('\n\t****** Coefficients, z-scores, p_values, & pre-transform min/max******')
    print('\n\t****** Coefficients, z-scores, p_values, & pre-transform min/max******',file=confmatrixfile)
    print('\n\t****** Coefficients, z-scores, p_values, & pre-transform min/max******',file=alltilesfile)
    outstring_head='\t{:20}    {:9} {:9} {:8}  {:8}    {:8}   {:5}    {:17}'
    outstring_rows='\t{:20} {:>9.5f} {:>10.4f} {:>8.3f} {:>10.3f}  {:>10.3f}   {:>7.5f}   {:>7.3f}/{:7.3f}'
    print(outstring_head.format('VARIABLE','COEFF','STD ERR','   Z','95% LL',
                                '95% UL','  p','PRETRANS MIN/MAX'))
    print(outstring_head.format('VARIABLE','COEFF','STD ERR','   Z','LWR LIM',
                                'UPR LIM',' p','PRETRANS MIN/MAX'),
                                 file=confmatrixfile)
    print(outstring_head.format('VARIABLE','COEFF','STD ERR','   Z','LWR LIM',
                                'UPR LIM','   p',' PRETRANS MIN/MAX'),
                                file=alltilesfile)
    for k in range(dfcoeffs.shape[0]):
# If the stderr is missing (Intercept), everything goes pear-shaped.
        try:
            lwrlim=dfcoeffs.loc[k,'Coeff']-st.norm.ppf(0.975)*dfcoeffs.loc[k,'stderr']
            uprlim=dfcoeffs.loc[k,'Coeff']+st.norm.ppf(0.975)*dfcoeffs.loc[k,'stderr']
        except:
            lwrlim=-99.999
            uprlim=-99.999
        if dfcoeffs.loc[k,'Coeff']== 0:
            lwrlim=-99.999
            uprlim=-99.999
        try:
            Xmin=dfXmin[dfcoeffs.loc[k,'Variable']]
            Xmax=dfXmax[dfcoeffs.loc[k,'Variable']]
        except:
            Xmin=-99.999
            Xmax=-99.999
        print(outstring_rows.format(dfcoeffs.loc[k,'Variable'],dfcoeffs.loc[k,'Coeff'],
            dfcoeffs.loc[k,'stderr'], dfcoeffs.loc[k,'Zscore'], lwrlim, uprlim, 
            dfcoeffs.loc[k,'pvalue'], Xmin, Xmax))
        print(outstring_rows.format(dfcoeffs.loc[k,'Variable'],dfcoeffs.loc[k,'Coeff'],
            dfcoeffs.loc[k,'stderr'], dfcoeffs.loc[k,'Zscore'], lwrlim, uprlim,
            dfcoeffs.loc[k,'pvalue'], Xmin, Xmax), file=confmatrixfile)
        print(outstring_rows.format(dfcoeffs.loc[k,'Variable'],dfcoeffs.loc[k,'Coeff'],
            dfcoeffs.loc[k,'stderr'], dfcoeffs.loc[k,'Zscore'], lwrlim, uprlim,
            dfcoeffs.loc[k,'pvalue'], Xmin, Xmax), file=alltilesfile)
    time_toc=time.time()
    print('\n******* Elapsed time for tile',file,' was {:.1f} mins. *******\n'.format((time_toc - time_tic) / 60))
    print('\n******* Elapsed time for tile',file,' was {:.1f} mins. *******\n'.format((time_toc - time_tic) / 60),
          file=confmatrixfile)
    print('\n******* Elapsed time for tile',file,' was {:.1f} mins. *******\n'.format((time_toc - time_tic) / 60),
          file=alltilesfile)
#################################################################################
# THE FOLLOWING CODE WAS SET UP FOR THE MLP CLASSIFIER AND HAS NEVER BEEN
# TESTED ON THE REGULARIZED REGESSION ANALYSIS.
# Now set up the reduced dataframe and output it. First attach the p(Bath)
# to the df.  Then subset the pulses whose p(Bath) exceeds the optimal
# threshold.
#    dfin['pbath']=Yprob[:,1]
#    dfhighp=dfin[dfin['pbath']>=opt_thresh.loc[0,'thresholds']]
#    dfhighp=dfhighp.drop('pbath',axis=1)
#    tempname=file.replace('_w_SBET_edge_scan_drctn_xy','')
#    dfhighp.to_csv(in_out_path+dfbathnameprefix+tempname.replace('df016_',''),
#                   index=False)
################################################################################
    confmatrixfile.close()
    tileoutpdf.close()
alltilesfile.close()
alloutpdf.close()