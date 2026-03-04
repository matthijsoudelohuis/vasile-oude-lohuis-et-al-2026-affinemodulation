#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from pylab import *
from scipy.sparse.linalg import svds
os.chdir('c:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from loaddata.get_data_folder import get_local_drive
from params import load_params
from utils.explorefigs import plot_PCA_gratings_3D,plot_PCA_gratings
from loaddata.session_info import *
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import *
from utils.pair_lib import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.RRRlib import LM,regress_out_behavior_modulation

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\TrialwiseModel\\')

#%% Plotting and parameters:
params  = load_params()
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_noiselevel=True)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_noiselevel=True)

#%%  Load data properly:
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=params['calciumversion'])
    # sessions[ises].respmat  /= sessions[ises].celldata['meanF'].to_numpy()[:,None] #convert to deconv/F0

#%%
# sessions = compute_tuning_wrapper(sessions)

#%%
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=50)

#%% Concatenate all cells:
celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
nCells                  = len(celldata)

#%%
def fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True):
    nPredictors = X.shape[1]
    predcat = np.array(np.repeat(['Stim','Mult','Add'],(1,nPredictors,nPredictors)))

    cvR2_models = np.full((4),np.nan)
    cvR2_preds  = np.full((4,nPredictors),np.nan)

    # Construct the design matrix
    A                           = np.column_stack([S[:,None],X * S[:,None], X])
    A                           = zscore(A,axis=0,nan_policy='omit')
    coefs, residuals, rank, s   = np.linalg.lstsq(A, y, rcond=None)    # Perform linear regression using least squares

    cvR2_models[0] = r2_score(y, A[:,np.isin(predcat,'Stim')] @ coefs[np.isin(predcat,'Stim')])
    cvR2_models[1] = r2_score(y, A[:,np.isin(predcat,['Stim','Mult'])] @ coefs[np.isin(predcat,['Stim','Mult'])])
    cvR2_models[2] = r2_score(y, A[:,np.isin(predcat,['Stim','Add'])] @ coefs[np.isin(predcat,['Stim','Add'])])
    cvR2_models[3] = r2_score(y, A @ coefs)

    for ipred in range(nPredictors):
        idx_pred_mult = ipred + 1
        idx_pred_add  = ipred + 1 + nPredictors

        A_shuf          = A.copy()
        A_shuf[:,idx_pred_mult] = np.random.permutation(A_shuf[:,idx_pred_mult])
        B       = np.linalg.lstsq(A_shuf, y, rcond=None)[0]    # Perform linear regression using least squares
        cvR2_preds[1,ipred] = cvR2_models[3] - r2_score(y,A_shuf @ B)
        
        A_shuf = A.copy()
        A_shuf[:,idx_pred_add] = np.random.permutation(A_shuf[:,idx_pred_add])
        B       = np.linalg.lstsq(A_shuf, y, rcond=None)[0]    # Perform linear regression using least squares
        cvR2_preds[2,ipred] = cvR2_models[3] - r2_score(y,A_shuf @ B)

        A_shuf[:,[idx_pred_mult,idx_pred_add]] = np.random.permutation(A_shuf[:,[idx_pred_mult,idx_pred_add]])
        B       = np.linalg.lstsq(A_shuf, y, rcond=None)[0]    # Perform linear regression using least squares
        cvR2_preds[3,ipred] = cvR2_models[3] - r2_score(y,A_shuf @ B)

    return cvR2_models,cvR2_preds

#%% Parameters for model fitting:
nbehavPCs               = 8
nvideoPCs               = 15

#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

nPredictors             = nbehavPCs + 1 # +1 for mean pop activity
# predlabels              = np.array([f'Behav_PC{i}' for i in range(nbehavPCs)] + ['MeanPopAct'])
predlabels              = np.array(['FF or FB'] + [f'Behav_PC{i}' for i in range(nbehavPCs)])
AffModels               = np.array(['Stim','Mult','Add','Both'])
nAffModels              = len(AffModels)

#initialize storage variables
cvR2_affine             = np.full((narealabelpairs,nAffModels,nCells),np.nan)
cvR2_preds              = np.full((narealabelpairs,nAffModels,nPredictors,nCells),np.nan)

pca                     = PCA(n_components=nbehavPCs)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing affine modulation models'):
    # zscore neural responses
    respdata        = zscore(sessions[ises].respmat, axis=1)
    # respdata        = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    #construct behavioral design matrix
    X       = np.stack((sessions[ises].respmat_videome,
                    sessions[ises].respmat_runspeed,
                    sessions[ises].respmat_pupilarea,
                    sessions[ises].respmat_pupilareaderiv,
                    sessions[ises].respmat_pupilx,
                    sessions[ises].respmat_pupily),axis=1)
    X       = np.column_stack((X,sessions[ises].respmat_videopc[:nvideoPCs,:].T))
    X       = zscore(X,axis=0,nan_policy='omit')
    si      = SimpleImputer() #impute missing values
    X       = si.fit_transform(X)

    X_p     = pca.fit_transform(X) #reduce dimensionality
    # #RRR to reduce dimensionality:
    # B_hat       = LM(respdata.T,X,lam=0)
    # U, s, V     = svds(B_hat,k=rank,which='LM')
    # X_p         = X @ U #project X onto the low rank subspace to get most predictive behavioral components

    #Get mean response per orientation (to predict trial by trial responses and multiplicative modulation)
    meanresp    = np.full_like(respdata,np.nan)
    trial_ori   = sessions[ises].trialdata['Orientation']
    for i,ori in enumerate(trial_ori.unique()):
        idx_T              = trial_ori == ori
        meanresp[:,idx_T] = np.nanmean(respdata[:,idx_T],axis=1)[:,None]

    # Fit affine modulation model for each arealabel pair (e.g. V1lab to PMunl for FF)
    # Compute mean population activity in area 1 (e.g. V1lab)
    # compute R2 of predicting responses in neurons in area 2 (e.g. PMunl) using 
    # behavioral PCs + mean pop activity in area 1
    for ialp,alp in enumerate(arealabelpairs):
        # idx_source_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        # idx_source_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]
        
        idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_N1) < params['minnneurons'] or len(idx_N2) < params['minnneurons']:
            continue
        
        meanpopact_N1          = np.nanmean(respdata[idx_N1,:],axis=0)
        meanpopact_N2          = np.nanmean(respdata[idx_N2,:],axis=0)

        for iN,N in enumerate(idx_N3):
            y       = respdata[N,:] 
            # X       = np.column_stack((X_p,meanpopact_N1,meanpopact_N2))
            # X       = np.column_stack((X_p,meanpopact_N1))
            X       = np.column_stack((meanpopact_N1,X_p))
            S       = meanresp[N,:]
            tempcvR2_models,tempcvR2_preds = fit_affine_FFFB(y,X,S,kfold=5,subtract_shuffle=True)

            idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][N])

            cvR2_affine[ialp,:,idx_ses] = tempcvR2_models
            cvR2_preds[ialp,:,:,idx_ses] = tempcvR2_preds


#%% Make schematic figure of affine model:
#Get good example cell, well described by both mult and add modulation:
ialp = 1
mult_diff = cvR2_affine[ialp,1] - cvR2_affine[ialp,0]
add_diff = cvR2_affine[ialp,2] - cvR2_affine[ialp,0]
add_diff = cvR2_affine[ialp,3] - cvR2_affine[ialp,0]
idx_examples = np.where(np.all((
                        cvR2_affine[ialp,0,:]>np.nanpercentile(cvR2_affine[ialp,0,:],80),
                        mult_diff>np.nanpercentile(mult_diff,80),
                        add_diff>np.nanpercentile(add_diff,80),
                       ),axis=0))[0]

example_cell      = np.random.choice(idx_examples,1)
# print(example_cell)
# example_cell = np.where(celldata['cell_id'] == 'LPE12223_2024_06_10_3_0012')[0]

ises = np.where(np.isin(sessiondata['session_id'], celldata['session_id'][example_cell]))[0][0]
idx_inses = np.where(np.isin(sessions[ises].celldata['cell_id'], celldata['cell_id'][example_cell]))[0]

trial_ori   = sessions[ises].trialdata['Orientation']
sortidx     = np.argsort(trial_ori)
nT          = len(trial_ori)

idx_source_N1  = np.where(np.all((sessions[ises].celldata['arealabel'] == 'V1lab',
                                        # sessions[ises].celldata['nearby']
                                        ),axis=0))[0]
meanpopact_N1          = np.nanmean(sessions[ises].respmat[idx_source_N1,:],axis=0)
meanpopact_N1 = zscore(meanpopact_N1)
#construct behavioral design matrix
X       = np.stack((sessions[ises].respmat_videome,
                sessions[ises].respmat_runspeed,
                sessions[ises].respmat_pupilarea,
                sessions[ises].respmat_pupilareaderiv,
                sessions[ises].respmat_pupilx,
                sessions[ises].respmat_pupily),axis=1)
X       = np.column_stack((X,sessions[ises].respmat_videopc[:nvideoPCs,:].T))
X       = zscore(X,axis=0,nan_policy='omit')
si      = SimpleImputer() #impute missing values
X       = si.fit_transform(X)
X_p     = pca.fit_transform(X) #reduce dimensionality

# zscore neural responses
y        = zscore(sessions[ises].respmat[idx_inses,:], axis=1)

#Get mean response per orientation (to predict trial by trial responses and multiplicative modulation)
S       = np.full_like(y,np.nan)
for i,ori in enumerate(trial_ori.unique()):
    idx_T              = trial_ori == ori
    S[:,idx_T] = np.nanmean(y[:,idx_T],axis=1)[:,None]

X       = np.column_stack((meanpopact_N1,X_p))

X_add = X
X_mult = X_add * S.T

y = y.T[sortidx]
X_add = X_add[sortidx,:]
X_mult = X_mult[sortidx,:]
S = S.T[sortidx]

subsampling_factor = 10
y = y[::subsampling_factor,:]
X_add = X_add[::subsampling_factor,:]
X_mult = X_mult[::subsampling_factor,:]
S = S[::subsampling_factor,:]

#  Construct the design matrix
A                           = np.column_stack([S,X_add, X_mult])
A                           = zscore(A,axis=0,nan_policy='omit')
coefs, residuals, rank, s   = np.linalg.lstsq(A, y, rcond=None)    # Perform linear regression using least squares
y_hat = A @ coefs

cmap = 'viridis'

# cmap = 'magma'

plt.rcParams.update({'figure.autolayout': False,
                        'image.aspect': 'auto'})

vmin,vmax = np.percentile(y,[5,95])
desired_width = 1 / np.shape(A)[1]
spacing = 0.05
xpos = 0.1
fig = plt.figure(figsize=(6*cm,8*cm))
datalabels = np.array(['Stim.','Add.','Mult.','Pred.','Actual'])
for i,data in enumerate([S,X_add,X_mult,y,y_hat]):
    ax = fig.add_subplot(111,position=[xpos,0.1,np.shape(data)[1]*desired_width,0.8])
    pcolor(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    ax.set_title(datalabels[i])
    if i==0:
        ax.text(-1,0.1,'Trials (subsampled and sorted by stimulus direction)',
                rotation=90,transform=ax.transAxes)
    xpos = xpos + np.shape(data)[1]*desired_width + spacing

my_savefig(fig,os.path.join(savedir,'ExampleNeurons'),'AffineModel_ExampleCell_cell%s' % (celldata['cell_id'][example_cell].to_numpy()[0]))

# %% 
clrs_arealabelpairs = ['#9933FF','#00CC99']
legendlabels        = ['FF','FB']
# legendlabels        = ['FF\n(V1->PM)','FB\n(PM->V1)']

#%% Show overall model performance: 
fig,axes = plt.subplots(1,2,figsize=(4*cm,4*cm),sharey=True,sharex=True) 
for ialp in range(narealabelpairs):
    ax = axes[ialp]
    ax.plot(np.arange(2),np.nanmean(cvR2_affine[ialp,[0,3],:],axis=(1)),marker=None,linewidth=1.3,color='k')
    ax.scatter(np.arange(2),np.nanmean(cvR2_affine[ialp,[0,3],:],axis=1),s=50,c=sns.color_palette('magma',2))
    ax.set_ylim([0,my_ceil(np.nanmean(cvR2_affine[1,-1,:])*1.1,2)])
    ax_nticks(ax, 3)
    ax.set_xlim([-0.2,1.2])
    ax.set_xticks(np.arange(2),labels=['Stim\nOnly','Full\nmodel'])
    ax.set_title(legendlabels[ialp])
axes[0].set_ylabel('Performance (R$^2$)')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'AffineModel_R2_StimvsAffine_Overall_%dsessions' % (nSessions))

#%% Show overall model performance: 
fig,axes = plt.subplots(1,1,figsize=(5*cm,4*cm)) 

ax = axes
idx_model = 3
ymean   = np.nanmean(cvR2_preds[:,idx_model,:,:],axis=(0,2))
yerr    = np.nanstd(cvR2_preds[:,idx_model,:,:],axis=(0,2)) / np.sqrt(np.sum(~np.isnan(cvR2_preds[0,idx_model,:,:]),axis=1))*10

ax.errorbar(np.arange(nPredictors),ymean,yerr,linestyle='', color='k',marker='o',
            linewidth=1,markersize=4)

ax.set_ylabel(u'Unique $\Delta R^2$')
ax_nticks(ax, 4)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticks(np.arange(nPredictors),labels=predlabels,rotation=45,ha='right')
my_savefig(fig,savedir,'AffineModel_uniqueR2_PredictorsOverall_%dsessions' % (nSessions))

#%% Show overall model performance as mean R2:
idx_N =    np.all((
            # celldata['gOSI']>0.5,
            celldata['gOSI']>0,
            # celldata['nearby'],
                ),axis=0)

fig,axes = plt.subplots(1,1,figsize=(5*cm,5*cm)) 
ax = axes
handles = []
for ialp,alp in enumerate(arealabelpairs):
    handles.append(ax.plot(np.arange(nAffModels),np.nanmean(cvR2_affine[ialp,:,idx_N],axis=0),marker=None,
            linestyle=['-','--'][ialp],linewidth=1.3,color='k')[0])
    ax.scatter(np.arange(nAffModels),np.nanmean(cvR2_affine[ialp,:,idx_N],axis=(0)),s=50,
            c=sns.color_palette('magma',nAffModels))
ax.set_ylim([0,my_ceil(np.nanmean(cvR2_affine[1,-1,idx_N]),2)])
ax.set_ylabel('Performance R2')
ax.legend(handles,legendlabels,fontsize=9,frameon=False,loc='lower right')
ax_nticks(ax, 5)
ax.set_xticks(np.arange(nAffModels),labels=AffModels)
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'AffineModel_R2_FF_vs_FB_%dsessions' % (nSessions), formats = ['png'])


# #%% Show overall model performance: 
# fig,axes = plt.subplots(4,1,figsize=(3,6),sharey=True,sharex=True)

# for ivar,var in enumerate(['Mult','Add']):
#     for ialp,alp in enumerate(arealabelpairs):
#         # ax = axes[ivar,ialp]
#         ax = axes[ivar*2 + ialp]
#         ymean   = np.nanmean(cvR2_preds[ialp,ivar+1,:,:],axis=1)
#         yerr    = np.nanstd(cvR2_preds[ialp,ivar+1,:,:],axis=1) / np.sqrt(np.sum(~np.isnan(cvR2_preds[ialp,ivar+1,:,:]),axis=1))*5
#         ax.errorbar(np.arange(nPredictors),ymean,yerr,linestyle='', color='k',marker='o',
#                     markersize=7,linewidth=2)
#         # ax.plot(np.arange(nPredictors),ymean,marker=None,
#         #         linewidth=1.3,color='k')
#         # shaded_error(np.arange(nPredictors),ymean,yerr,color='black',alpha=0.2,ax=ax)
#         ax.set_ylim([0,my_ceil(np.nanmax(np.nanmean(cvR2_preds,axis=(1,3)).flatten()),2)])
#         ax.set_ylabel(u'$\Delta R^2$')
#         ax_nticks(ax, 3)
#         ax.set_title(f'{legendlabels[ialp]} - {var}')
#         ax.set_xticks(np.arange(nPredictors),labels=predlabels,rotation=45,ha='right')

# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# ax.set_xticks(np.arange(nPredictors),labels=predlabels,rotation=45,ha='right')
# # my_savefig(fig,savedir,'AffineModel_R2_MultAddSep_PredictorsOverall_%dsessions' % (nSessions), formats = ['png'])


# #%% Show overall model performance as histograms:
# fig,axes = plt.subplots(1,4,figsize=(12,3),sharey=True,sharex=True)
# for imodel in range(nAffModels):
#     ax = axes[imodel]
#     for ialp,alp in enumerate(arealabelpairs):
#         sns.histplot(cvR2_affine[ialp,imodel,:],bins=np.linspace(-0.1,1,50),element='step',stat='probability',
#                  color=clrs_arealabelpairs[ialp],fill=False,ax=ax)
#     ax.set_xlabel('R2')
#     ax.set_title(AffModels[imodel])
#     ax.legend(legendlabels,fontsize=9,frameon=False)
# sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
# # my_savefig(fig,savedir,'AffineModel_Hist_FF_FB_R2_%dsessions' % (nSessions), formats = ['png'])


#%% 
fig,axes = plt.subplots(nPredictors,2,figsize=(4,nPredictors*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    for ipred in range(nPredictors):
        ax = axes[ipred,imodel]
        idx_N =  np.all((
                celldata['gOSI']>0,
                # celldata['gOSI']>0.2,
                # celldata['nearby'],
                    ),axis=0)
        ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)
        # yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(~np.isnan(cvR2_preds[:,model,ipred,idx_N]),axis=1))
        
        # confidence interval
        d = cvR2_preds[0,model,ipred,idx_N]
        yerror1 = np.nanmean(d) - stats.t.interval(0.99, df=len(d)-1, loc=np.nanmean(d), scale=np.nanstd(d, ddof=1) / np.sqrt(len(d)))
        d = cvR2_preds[1,model,ipred,idx_N]
        yerror2 = np.nanmean(d) - stats.t.interval(0.99, df=len(d)-1, loc=np.nanmean(d), scale=np.nanstd(d, ddof=1) / np.sqrt(len(d)))
        yerror = np.abs(np.array([yerror1,yerror2]).T)
        
        ax.bar([0,1],height=ymean,yerr=0,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
        ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                    linewidth=4)
        ax.set_xticks([0,1],labels=legendlabels)
        h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
                            cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
        p = p * narealabelpairs
        add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
        ax_nticks(ax, 3)
        if imodel == 0:
            ax.set_ylabel(u'$\Delta R^2$')
        if ipred ==0:
            ax.set_title(AffModels[model])
        # ax.legend(legendlabels,fontsize=9,frameon=False)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Affinemodulation_R2_Allpredictors_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% Show for the interarea modulation only:
ipred = 0
fig,axes = plt.subplots(1,2,figsize=(4.5*cm,4*cm),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    # for ipred in range(nPredictors):
    ax = axes[imodel]
    idx_N = np.all((
            rangeresp>params['minrangeresp'],
                ),axis=0)
    ymean = np.nanmean(cvR2_preds[:,model,ipred,idx_N],axis=1)

    # confidence interval
    d = cvR2_preds[0,model,ipred,idx_N]
    yerror1 = np.nanmean(d) - stats.t.interval(0.99, df=len(d)-1, loc=np.nanmean(d), scale=np.nanstd(d, ddof=1) / np.sqrt(len(d)))
    d = cvR2_preds[1,model,ipred,idx_N]
    yerror2 = np.nanmean(d) - stats.t.interval(0.99, df=len(d)-1, loc=np.nanmean(d), scale=np.nanstd(d, ddof=1) / np.sqrt(len(d)))
    yerror = np.abs(np.array([yerror1,yerror2]).T)
    
    # yerror = np.nanstd(cvR2_preds[:,model,ipred,idx_N],axis=1) / np.sqrt(np.sum(idx_N)/2)
    ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=1)
    ax.set_xticks([0,1],labels=legendlabels)
    h,p = stats.ttest_ind(cvR2_preds[0,model,ipred,idx_N],
                        cvR2_preds[1,model,ipred,idx_N],nan_policy='omit')
    p = p * narealabelpairs
    add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0,fontsize=7)
    ax_nticks(ax, 4)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks,yticks*1000)
    if imodel == 0:
        ax.set_ylabel(u'Unique $\Delta R^2$')
        ax.text(-0.1,1,u'x10$^-3$',transform=ax.transAxes)

    ax.set_title(AffModels[model])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_barplot_GR%dsessions' % (nSessions))

#%% make a violinplot instead of a barplot
# legendlabels        = ['FF','FB']
# ipred               = -1
# upperclipvalue      = my_ceil(np.nanpercentile(cvR2_preds[:,1,-1,:],95),2)
# df = pd.DataFrame({'deltaR2': [], 'direction': [], 'modulation': []})
# for ialp,arealabel in enumerate(legendlabels):
#     for imodulation,modulation in enumerate(['Mult','Add']):
#         df = pd.concat((df,pd.DataFrame({'deltaR2': cvR2_preds[ialp,imodulation+1,ipred,:], 'direction': np.repeat(arealabel,nCells), 'modulation': np.repeat(modulation,nCells)})))
# df.dropna(inplace=True)
# df['deltaR2'].clip(lower=-0.0025,upper=upperclipvalue,inplace=True)
# fig,ax = plt.subplots(1,1,figsize=(4,4))
# sns.violinplot(data=df, x="modulation", y="deltaR2", hue="direction", hue_order=legendlabels, 
#                linewidth=1,palette=clrs_arealabelpairs,split=True, inner="quart",ax=ax)
# ax.axhline(upperclipvalue,linestyle='--',color='k',linewidth=0.5)
# ax.text(0.8,upperclipvalue+0.0005,'clip')
# ax_nticks(ax, 7)
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_Affinemodulation_dR2_violinplot_%dsessions' % (nSessions), formats = ['png'])

#%% Is the amount of multiplicative and additive modulation related across neurons?

# #%% Show for the interarea modulation only:
# ipred = -1
# fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharey='row',sharex=True)
# for ialp,alp in enumerate(legendlabels): #Mult and Add
#     ax = axes[ialp]
#     idx_N = np.all((
#             celldata['gOSI']>0,
#                 ),axis=0)
#     x = cvR2_preds[ialp,1,ipred,idx_N]
#     y = cvR2_preds[ialp,2,ipred,idx_N]

#     x = x[~np.isnan(y)]
#     y = y[~np.isnan(y)]
#     sns.scatterplot(x=x,y=y,s=5,alpha=0.2,color=clrs_arealabelpairs[ialp],ax=ax)
    
#     # sns.regplot(x=x,y=y,x_ci='sd',
#     #             scatter_kws={'s':2,'alpha':0.2,'color': clrs_arealabelpairs[ialp]},ax=ax)
#     ax.set_xlim(np.nanpercentile([x],[0,99.8]))
#     ax.set_ylim(np.nanpercentile([y],[0,99.8]))
#     ax_nticks(ax, 3)
#     ax.set_xlabel(u'Multiplicative $\Delta R^2$')
#     ax.set_ylabel(u'Additive  $\Delta R^2$')
#     ax.set_title(legendlabels[ialp],fontsize=11)
#     ax.text(0.8,0.8,'r = %.2f' % (stats.pearsonr(x,y)[0]),transform=ax.transAxes,fontsize=9) #print(stats.pearsonr(x,y)[0])
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'Corr_Mult_Add_uniqueR2_FF_FB_GR%dsessions' % (nSessions), formats = ['png'])




#%% Show the same figure but for behavioral predictors only
arealabels = ['PM','V1']
ipred = np.arange(1,nbehavPCs)
fig,axes = plt.subplots(1,2,figsize=(3,1*2),sharey='row',sharex=True)
for imodel,model in enumerate([1,2]): #Mult and Add
    ax = axes[imodel]
    idx_N =  np.all((
            celldata['gOSI']>0,
            # celldata['gOSI']>0.2,
            # celldata['nearby'],
                ),axis=0)
    ymean = np.nanmean(cvR2_preds[np.ix_(range(narealabelpairs),[model],ipred,np.where(idx_N)[0])],axis=(1,3))
    ymean = np.nansum(ymean,axis=1)
    yerror = np.nanstd(cvR2_preds[np.ix_(range(narealabelpairs),[model],ipred,np.where(idx_N)[0])],axis=(1,2,3)) / np.sqrt(np.sum(idx_N)/10)
    ax.bar([0,1],height=ymean,yerr=yerror,color=clrs_arealabelpairs)#,errorbar=('ci', 95))
    ax.errorbar([0,1],y=ymean,yerr=yerror,linestyle='', color='k',
                linewidth=4)
    ax.set_xticks([0,1],labels=arealabels)
    xdata = np.nanmean(cvR2_preds[np.ix_([0],[model],ipred,np.where(idx_N)[0])],axis=(1,2)).squeeze()
    ydata = np.nanmean(cvR2_preds[np.ix_([1],[model],ipred,np.where(idx_N)[0])],axis=(1,2)).squeeze()
    h,p = stats.ttest_ind(xdata,ydata,nan_policy='omit')
    p = p * narealabelpairs
    add_stat_annotation(ax, 0.2, 0.8,ymean.max()*1.1, p, h=0)
    ax_nticks(ax, 3)
    if imodel == 0:
        ax.set_ylabel(u'$\Delta R^2$')
    ax.set_title(AffModels[model],fontsize=11)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Behavior_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% Is the effect present in all the sessions?
ipred = 0
datamat = np.full((narealabelpairs,2,nSessions),np.nan)
ncellmat = np.full((narealabelpairs,2,nSessions),np.nan)
for ises in range(nSessions):
    # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'])
    idx_ses =  np.all((
            np.isin(celldata['session_id'],sessions[ises].celldata['session_id']),
            # celldata['gOSI']>0.5,
            # celldata['gOSI']>0.2,
            # ~celldata['nearby'],
                ),axis=0)
    for ialp in range(narealabelpairs):
        for imodtype in range(2):
            ntargetcells = np.sum(~np.isnan(cvR2_preds[ialp,imodtype+1,ipred,idx_ses]))
            ncellmat[ialp,imodtype,ises] = ntargetcells
            datamat[ialp,imodtype,ises] = np.nanmean(cvR2_preds[ialp,imodtype+1,ipred,idx_ses])
            # datamat[ialp,imodtype,ises] = np.nanmedian(cvR2_preds[ialp,imodtype+1,ipred,idx_ses])

#%% Showing for different sessions and normalizing for mean for each session
#Normalizing for the mean across modulation types
datamatnorm = datamat
datamatnorm = datamat - np.mean(datamat,axis=(1),keepdims=True)
datamatnorm = datamatnorm + np.mean(datamat,axis=(0,1),keepdims=True)
datamatnorm[:,:,8] = np.nan

fig,axes = plt.subplots(1,2,figsize=(4*cm,4*cm),sharey='row',sharex=True)
ax = axes[0]
# sns.lineplot(datamatnorm[:,0,:],linewidth=1,ax=ax,legend=False,linestyle='-',
#                 palette=sns.color_palette("dark",nSessions))
# sns.lineplot(datamatnorm[:,0,:],linewidth=1,ax=ax,legend=False,linestyle='-',
#                 linecolor='k')
ax.plot(datamatnorm[:,0,:],color='k')
ax.scatter(np.zeros(nSessions),datamatnorm[0,0,:],s=7,color=clrs_arealabelpairs[0])
ax.scatter(np.ones(nSessions),datamatnorm[1,0,:],s=7,color=clrs_arealabelpairs[1])
p = stats.ttest_rel(datamatnorm[0,0,:],datamatnorm[1,0,:],nan_policy='omit')[1]
# p = stats.ttest_rel(datamat[0,0,:],datamat[1,0,:],nan_policy='omit')[1]
add_stat_annotation(ax, 0.2, 0.8,0.013, p, h=0,fontsize=9)
ax.set_title('Multiplicative',fontsize=6)
ax.set_xticks([0,1],labels=legendlabels)
ax.set_ylabel(u'Unique $\Delta R^2$ (norm)')
ax.set_ylim([0,my_ceil(np.nanmax(datamatnorm)*1.1,3)])
ax_nticks(ax, 4)
ax = axes[1]
ax.plot(datamatnorm[:,1,:],color='k')
ax.scatter(np.zeros(nSessions),datamatnorm[0,1,:],s=7,color=clrs_arealabelpairs[0])
ax.scatter(np.ones(nSessions),datamatnorm[1,1,:],s=7,color=clrs_arealabelpairs[1])

# sns.lineplot(datamatnorm[:,1,:],linewidth=1,ax=ax,legend=False,linestyle='-',
#                 palette=sns.color_palette("dark",nSessions))
p = stats.ttest_rel(datamatnorm[0,1,:],datamatnorm[1,1,:],nan_policy='omit')[1]
add_stat_annotation(ax, 0.2, 0.8,0.013, p, h=0,fontsize=9)
ax.set_title('Additive',fontsize=6)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_affinemodulation_sessionplot_GR%dsessions' % (nSessions))

#%% Is the variability explained by behavior and population rate in the same neurons?
ibehavpred = 0
ipopactpred = -1
fig,axes = plt.subplots(2,2,figsize=(5*cm,5*cm),sharey=True,sharex=True)
for ialp in range(narealabelpairs):
    for imodtype in range(2):
        ax = axes[ialp,imodtype]
        # x = cvR2_preds[ialp,imodtype+1,ibehavpred,:]
        x = np.nansum(cvR2_preds[ialp,imodtype+1,range(nbehavPCs),:],axis=0)
        # x = np.nansum(cvR2_preds[ialp,imodtype+1,[0,1],:],axis=0)
        # y = np.nansum(cvR2_preds[ialp,imodtype+1,[2,3,4],:],axis=0)
        
        y = cvR2_preds[ialp,imodtype+1,ipopactpred,:]
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        sns.scatterplot(x=x,y=y,s=4,alpha=0.3,color=clrs_arealabelpairs[ialp],ax=ax)
       
        # sns.regplot(x=x,y=y,x_ci='sd',
        #             scatter_kws={'s':2,'alpha':0.2,'color': clrs_arealabelpairs[ialp]},ax=ax)
        ax.set_xlim(np.nanpercentile([x],[0,99.9]))
        ax.set_ylim(np.nanpercentile([y],[0,99.9]))
        ax_nticks(ax, 3)
        ax.set_xlabel(u'Behavior ΔR2')
        ax.set_ylabel(u'Cross area modulation ΔR2')
        ax.set_title(AffModels[imodtype+1] + ' ' + legendlabels[ialp])
        ax.text(0.6,0.6,'r = %.2f' % (stats.pearsonr(x,y)[0]),transform=ax.transAxes)

        # ax.text(0.8,0.8,'r = %.2f' % (stats.spearmanr(x,y)[0]),transform=ax.transAxes,fontsize=9) #print(stats.pearsonr(x,y)[0])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'AffMod_Corr_Behavior_Poprate_GR%dsessions' % (nSessions))

#%% This is not the case for behavioral variables amongst each other:
corrs = []
for ialp in range(narealabelpairs):
    for imodtype in range(2):
        x = np.nansum(cvR2_preds[ialp,imodtype+1,range(nbehavPCs//2),:],axis=0)
        y = np.nansum(cvR2_preds[ialp,imodtype+1,range(nbehavPCs//2,nbehavPCs),:],axis=0)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        corrs.append(stats.pearsonr(x,y)[0])
print('Correlation between behavior variables: %.2f' % np.nanmean(corrs))

#%% ## Show that the amount of multiplicative and additive modulation varies with orientation selectivity
# for behavioral variables
fig,axes = plt.subplots(1,2,figsize=(6,3),sharey=True,sharex=True)
x = celldata['gOSI']
# idx_predictor = [0]
idx_predictor = np.arange(1,nbehavPCs)
ydata = np.nanmean(cvR2_preds[0,1,idx_predictor,:],axis=(0))
sns.regplot(x=x,y=ydata,x_ci='sd',scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[0]},
            line_kws={'color': clrs_arealabelpairs[0],'linewidth':3,'linestyle':'-'},ax=axes[0])
ydata = np.nanmean(cvR2_preds[1,1,idx_predictor,:],axis=(0))
sns.regplot(x=x,y=ydata,ci=99,scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[1]},
            line_kws={'color': clrs_arealabelpairs[1],'linewidth':3,'linestyle':'-'},ax=axes[0])
axes[0].set_ylim(np.nanpercentile(ydata,[0.5,98]))
axes[0].set_title('Multiplicative')
axes[0].set_ylabel('Delta R2')
axes[0].set_xlim([0,1])
ax_nticks(axes[0], 3)
ydata = np.nanmean(cvR2_preds[0,2,idx_predictor,:],axis=(0))
sns.regplot(x=x,y=ydata,x_ci='sd',scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[0]},
            line_kws={'color': clrs_arealabelpairs[0],'linewidth':3,'linestyle':'-'},ax=axes[1])
ydata = np.nanmean(cvR2_preds[1,2,idx_predictor,:],axis=(0))
sns.regplot(x=x,y=ydata,ci=99,scatter_kws={'s':2,'alpha':0.1,'color': clrs_arealabelpairs[1]},
            line_kws={'color': clrs_arealabelpairs[1],'linewidth':3,'linestyle':'-'},ax=axes[1])
axes[1].set_ylim(np.nanpercentile(cvR2_preds[:,2,idx_predictor,:],[0.5,98]))
axes[1].set_title('Additive')
axes[1].set_xlim([0,1])
ax_nticks(axes[1], 3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Behav_affineR2_control_gOSI_GR%dsessions' % (nSessions), formats = ['png'])

#%% Test if the multiplitivate and additive effect persists even if you control for orientation tuning strength
idx_predictor = np.arange(1,nbehavPCs)
FF = np.any(~np.isnan(cvR2_preds[0,:,:,:]),axis=(0,1))
FB = np.any(~np.isnan(cvR2_preds[1,:,:,:]),axis=(0,1))
arealabel = np.repeat('None',nCells) 
arealabel[np.where(FF & ~FB)[0]] = 'FF'
arealabel[np.where(~FF & FB)[0]] = 'FB'

df = pd.DataFrame({'gOSI':    celldata['gOSI'],
                   'session_id': celldata['session_id'],
                   'Mult':  np.nanmean(cvR2_preds[:,1,idx_predictor,:],axis=(0,1)),
                   'Add':    np.nanmean(cvR2_preds[:,2,idx_predictor,:],axis=(0,1)),
                   'AreaLabel': arealabel,
                   })
df.dropna(inplace=True)

#%% Test mixed effects linear model for multiplicative modulation:
# Does gOSI explain variance in multiplicative and additive modulation for behavioral modulation differently?
model     = smf.mixedlm("Mult ~  C(AreaLabel, Treatment('FF')) * gOSI", data=df,groups=df["session_id"])
result    = model.fit(reml=False)
print(result.summary())
# 0.000  0.002

#%% test for additive modulation
model     = smf.mixedlm("Add ~  C(AreaLabel, Treatment('FF')) * gOSI", data=df,groups=df["session_id"])
result    = model.fit(reml=False)
print(result.summary())
