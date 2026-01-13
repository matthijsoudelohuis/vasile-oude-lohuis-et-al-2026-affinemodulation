#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import norm
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score
from scipy.stats import linregress,binned_statistic,pearsonr,spearmanr
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# os.chdir('e:\\Python\\molanalysis')
# os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')
os.chdir('c:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance,value_matching,filter_nearlabeled
from utils.plot_lib import * #get all the fixed color schemes
from loaddata.session_info import assign_layer,assign_layer2
from utils.RRRlib import regress_out_behavior_modulation

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\SplitTrials\\')

cm = 1/2.54  # centimeters in inches
plt.rcParams.update({'font.size': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.titlesize': 8,
                     'axes.labelpad': 1, 'ytick.major.pad': 1, 'xtick.major.pad': 1})


#%% 
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

clrs_arealabelpairs         = get_clr_arealabelpairs(arealabelpairs)
clrs_arealabels_low_high    = get_clr_area_low_high()  # PMlab-PMunl-V1unl
minrangeresp                = 0.04

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%%  Load data properly:
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=calciumversion)

#%% Compute Tuning Metrics (gOSI, gDSI etc.)
sessions = compute_tuning_wrapper(sessions)

#%%
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=50)

#%%  #assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 

    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 

# #%% 
# arealayers = np.array(['V1L2/3','PML2/3','PML5'])
# narealayers = len(arealayers)
# maxnoiselevel = 20
# celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
# clrs = ['black','red']
# fig,axes = plt.subplots(1,narealayers,figsize=(narealayers*3,3),sharey=True,sharex=True)

# for ial,al in enumerate(arealayers):
#     ax = axes[ial]
#     idx_N              = np.all((
#                                 celldata['noise_level']<maxnoiselevel,
#                                 celldata['arealayer']==al,
#                                 celldata['nearby'],
#                                     ),axis=0)

#     sns.histplot(data=celldata[idx_N],x='pop_coupling',hue='arealayerlabel',color='green',element="step",stat="density", 
#                 common_norm=False,fill=False,bins=np.linspace(-0.3,1,1000),cumulative=True,ax=ax,palette=clrs,legend=False)
#     ax.set_title(al)
# sns.despine(fig=fig, top=True, right=True,offset=0)


#%% For every session remove behavior related variability:
rank_behavout = 3
# maxnoiselevel = 20

#%%
for ises in range(nSessions):
    # Convert response_matrix and orientations_vector to numpy arrays
    # response_matrix         = np.array(response_matrix)
    conditions_vector       = np.array(sessions[ises].trialdata['stimCond'])
    conditions              = np.sort(np.unique(conditions_vector))
    C                       = len(conditions)

    resp_mean       = sessions[ises].respmat.copy()
    # resp_res        = sessions[ises].respmat.copy()

    for iC,cond in enumerate(conditions):
        tempmean                            = np.nanmean(sessions[ises].respmat[:,conditions_vector==cond],axis=1)
        # resp_mean[:,iC]                     = tempmean
        resp_mean[:,conditions_vector==cond] = tempmean[:,np.newaxis]

    Y               = sessions[ises].respmat.copy()
    Y               = Y - resp_mean
    [Y_orig,Y_hat,Y_out,rank,ev] = regress_out_behavior_modulation(sessions[ises],X=None,Y=Y.T,
                                nvideoPCs = 30,rank=rank_behavout,lam=0,perCond=True,kfold = 5)
    # print(ev)
    sessions[ises].respmat_behavout = Y_out.T + resp_mean

# plt.imshow(sessions[ises].respmat,cmap='RdBu_r',vmin=0,vmax=100)
# plt.imshow(sessions[ises].respmat_behavout,cmap='RdBu_r',vmin=0,vmax=100)

# %%
 #####  ####### ### #       #          ####### ######  ###    #    #        #####  
#     #    #     #  #       #             #    #     #  #    # #   #       #     # 
#          #     #  #       #             #    #     #  #   #   #  #       #       
 #####     #     #  #       #             #    ######   #  #     # #        #####  
      #    #     #  #       #             #    #   #    #  ####### #             # 
#     #    #     #  #       #             #    #    #   #  #     # #       #     # 
 #####     #    ### ####### #######       #    #     # ### #     # #######  #####  

#%% 
# respmat_videoME = []
# for ises in range(nSessions):   
#     respmat_videoME.append(list(sessions[ises].respmat_videome))
maxvideome              = 0.2
maxrunspeed             = 0.5
maxnoiselevel           = 20

respmat_videoME = np.array([])
respmat_runspeed = np.array([])
for ises in range(nSessions):
    ses_videoME = sessions[ises].respmat_videome - np.nanpercentile(sessions[ises].respmat_videome,0)
    ses_videoME = ses_videoME/np.nanpercentile(ses_videoME,100)
    respmat_videoME = np.append(respmat_videoME,ses_videoME)
    respmat_runspeed = np.append(respmat_runspeed,sessions[ises].respmat_runspeed)

idx_T_still = np.logical_and(respmat_videoME < maxvideome,
                            respmat_runspeed < maxrunspeed)
fig,axes = plt.subplots(2,2,figsize=(3,3))
ax = axes[0,0]
sns.histplot(respmat_runspeed[idx_T_still],bins=np.linspace(-1,60,200),color='black',
             element='step',stat='count',fill=True,ax=ax)
sns.histplot(respmat_runspeed[~idx_T_still],bins=np.linspace(-1,60,200),color='grey',
             element='step',stat='count',fill=True,ax=ax)

ax = axes[1,1]
# sns.histplot(respmat_videoME,bins=np.linspace(0,1,50),element='step',stat='probability',fill=False,ax=ax)
sns.histplot(y=respmat_videoME[idx_T_still],bins=np.linspace(0,1,200),color='black',
             element='step',stat='count',fill=True,ax=ax)
sns.histplot(y=respmat_videoME[~idx_T_still],bins=np.linspace(0,1,200),color='grey',
             element='step',stat='count',fill=True,ax=ax)
ax.text(0.4,0.8,'Still (Included)',color='black',transform=ax.transAxes,fontsize=9)
ax.text(0.4,0.7,'Moving (Excluded)',color='grey',transform=ax.transAxes,fontsize=9)


ax = axes[1,0]
# sns.scatterplot(ax=ax,x=respmat_videoME,y=respmat_runspeed,alpha=0.25,s=4,hue=idx_T_still,
sns.scatterplot(ax=ax,x=respmat_runspeed,y=respmat_videoME,alpha=0.25,s=4,hue=idx_T_still,
                palette=['grey','black'],legend=False)
# ax.legend(['Still','Moving'],fontsize=9,frameon=False)
ax.text(0.3,0.6,'n = %d/%d \nstill trials \n(%.1f%%)' % (np.sum(idx_T_still),len(idx_T_still),np.sum(idx_T_still)/len(idx_T_still)*100),
        transform=ax.transAxes,fontsize=6)
ax.set_ylabel('Video ME (norm.)')
ax.set_xlabel('Running speed (cm/s)')
# plt.tight_layout()
axes[0,1].axis('off')
sns.despine(fig=fig, top=True, right=True,offset=0)
my_savefig(fig,savedir,'StillTrials_Selection')


#%% 
ises = 0
fig, axes = plt.subplots(1,1,figsize=(6,2.5))
ax = axes
idx_N_FF              = np.where(np.all((sessions[ises].celldata['arealabel'] == 'V1lab',),axis=0))[0]
idx_N_FB              = np.where(np.all((sessions[ises].celldata['arealabel'] == 'PMlab',),axis=0))[0]
idx_T_still           = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                            sessions[ises].respmat_runspeed < maxrunspeed)
respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

meanpopact_FF          = np.nanmean(respdata[idx_N_FF,:],axis=0)
meanpopact_FB          = np.nanmean(respdata[idx_N_FB,:],axis=0)
ax.plot(meanpopact_FF[idx_T_still],color=clrs_arealabelpairs[0],linewidth=0.4)
ax.plot(meanpopact_FB[idx_T_still],color=clrs_arealabelpairs[1],linewidth=0.4)
ax.set_xlim([0,500])
ax.set_ylim(np.percentile([meanpopact_FF[idx_T_still][:500],meanpopact_FB[idx_T_still][:500]],[0,100]))
ax.set_xlabel('Trials',fontsize=10)
ax.set_ylabel('Population activity (deconv/dF0)',fontsize=10)
ax.set_title('Population activity',fontsize=13)
ax.text(0.1,0.8,'r = %1.2f, p = %1.2f' % (pearsonr(meanpopact_FF[idx_T_still],meanpopact_FB[idx_T_still])[0],pearsonr(meanpopact_FF[idx_T_still],meanpopact_FB[idx_T_still])[1]),transform=ax.transAxes,fontsize=8)
ax_nticks(ax,4)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
# my_savefig(fig,savedir,'PopulationActivity_Trials_FF_FB', formats = ['png'])

#%% Correlation matrix between the different populations
arealabelpairs  = ['V1unl','V1lab','PMunl','PMlab']
narealabelpairs = len(arealabelpairs)
corrmat = np.full((narealabelpairs,narealabelpairs,nSessions),np.nan)
for ises in range(nSessions):
    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    # idx_T_still = np.ones(np.sum(idx_T_still),dtype=bool)
    datamat = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    nsampleneurons = 1000
    for ial,alp in enumerate(arealabelpairs):
        nsampleneurons = np.min([nsampleneurons,
                                 len(np.where(np.all((sessions[ises].celldata['arealabel'] == alp,
                                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                      sessions[ises].celldata['nearby'],
                                                      ),axis=0))[0])])
    
    for ial,alp in enumerate(arealabelpairs):
        idx_N               = np.where(np.all((sessions[ises].celldata['arealabel'] == alp,),axis=0))[0]
        idx_N               = np.where(np.all((sessions[ises].celldata['arealabel'] == alp,
                                            sessions[ises].celldata['noise_level']<maxnoiselevel),axis=0))[0]
        idx_N               = np.random.choice(idx_N, size=nsampleneurons, replace=False)
        datamat[ial,:]      = np.nanmean(sessions[ises].respmat[np.ix_(idx_N,idx_T_still)],axis=0)
    corrmat[:,:,ises] = np.corrcoef(datamat)

fig, axes = plt.subplots(1,1,figsize=(6,2.5))
ax = axes
ax.imshow(np.nanmean(corrmat,axis=2),cmap='RdBu_r',vmin=-.3,vmax=.3)
ax.set_xticks(range(narealabelpairs))
ax.set_xticklabels(arealabelpairs)
ax.set_yticks(range(narealabelpairs))
ax.set_yticklabels(arealabelpairs)
ax.set_title('Correlation matrix',fontsize=13)
cbar = fig.colorbar(ax.imshow(np.nanmean(corrmat,axis=2),cmap='RdBu_r',vmin=-.3,vmax=.3), ax=ax)
cbar.set_label('Correlation', rotation=270, labelpad=10)


#%% 
ises = 6
maxrunspeed = 0.5
maxvideome = 0.2

idx_T_still = np.where(np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed))[0]

arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']
# axislabels  = ['V1unl','V1lab','PMunl','PMlab']
# axislabels = arealabeled_to_figlabels(alp.split('-')[0])

vscale      = 0.015
# vscale      = 2
# fig,axes    = plt.subplots(1,2,figsize=(6,3))
# maxnoiselevel = 20
# fig,axes = plt.subplots(2,2,figsize=(6,6))
fig,axes = plt.subplots(2,1,figsize=(3.5,6))
for ialp,alp in enumerate(arealabelpairs):
    respdata        = sessions[ises].respmat
    # respdata        = zscore(sessions[ises].respmat, axis=1)
    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]
    idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]

    subsampleneurons    = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
    idx_source_N1       = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
    idx_source_N2       = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

    meanpopact_N1       = np.nanmean(respdata[np.ix_(idx_source_N1,idx_T_still)],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)
    meanpopact_N2       = np.nanmean(respdata[np.ix_(idx_source_N2,idx_T_still)],axis=0)

    vmin,vmax = np.percentile([meanpopact_N1 - meanpopact_N2],[5,95])
    # vmax = np.percentile([meanpopact_N2,meanpopact_N1],[99.5,99.5])
    ax=axes[ialp]
    ax.scatter(meanpopact_N2,meanpopact_N1,c=meanpopact_N1 - meanpopact_N2,s=0.2,cmap='coolwarm',vmin=vmin,vmax=vmax)
    # ax.scatter(meanpopact_N2,meanpopact_N1,c=meanpopact_N1 / meanpopact_N2,s=0.2,cmap='coolwarm',vmin=-vscale,vmax=vscale)

    ax.text(0.1,0.8,'%s > %s' % (arealabeled_to_figlabels([alp.split('-')[0]]),arealabeled_to_figlabels([alp.split('-')[1]])),transform=ax.transAxes)
    ax.text(0.7,0.2,'%s < %s' % (arealabeled_to_figlabels([alp.split('-')[0]]),arealabeled_to_figlabels([alp.split('-')[1]])),transform=ax.transAxes)
    # ax.text(0.7,0.2,'%s < %s' % (arealabeled_to_figlabels([alp.split('-')[0]]),transform=ax.transAxes)
    ax.plot([-1,1],[-1,1],color='k',linewidth=0.2)
    ax.set_xlim(np.percentile([meanpopact_N2,meanpopact_N1],[0.1,99.5]))
    ax.set_ylim(np.percentile([meanpopact_N2,meanpopact_N1],[0.1,99.5]))
    ax.set_xlabel('Mean %s activity (events/F0) ' % arealabeled_to_figlabels([alp.split('-')[1]]))
    ax.set_ylabel('Mean %s activity (events/F0) ' % arealabeled_to_figlabels([alp.split('-')[0]]))
    ax_nticks(ax, 3)
    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True,offset=1)
my_savefig(fig,savedir,'Diff_LabUnl_%s' % sessiondata['session_id'][ises])

#%% Show the distribution of mean activity of the labeled population and the selection of the
# extreme percentiles:
arealabelpairs  = ['V1lab','PMlab']
arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']

# clrs_arealabelpairs = get_clr_arealabelpairs(arealabelpairs)
legendlabels        = ['FF','FB']
clrs_arealabels_low_high = get_clr_area_low_high()

respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

perc = 25
step  = 0.001
fig, axes = plt.subplots(1,2,figsize=(6,3))
for ial,alp in enumerate(arealabelpairs):
    ax = axes[ial]
    idx_N1              = np.where(np.all((
                                        sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                        ),axis=0))[0]
    meanpopact_N1          = np.nanmean(respdata[idx_N1,:],axis=0)
    idx_N2              = np.where(np.all((
                                        sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                        ),axis=0))[0]
    meanpopact_N2          = np.nanmean(respdata[idx_N2,:],axis=0)

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    # meanpopact = meanpopact_N1
    meanpopact = meanpopact_N1 - meanpopact_N2

    allbins = np.arange(np.percentile(meanpopact[idx_T_still],0),np.percentile(meanpopact[idx_T_still],100),step)
    bins = allbins[allbins <= np.percentile(meanpopact[idx_T_still],perc)]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins[bins<np.percentile(meanpopact[idx_T_still],perc)],color=clrs_arealabels_low_high[ial,0],edgecolor='none')
    bins = allbins[np.logical_and(allbins >= np.percentile(meanpopact[idx_T_still],perc), allbins <= np.percentile(meanpopact[idx_T_still],100-perc))]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color='grey',edgecolor='none')
    bins = allbins[allbins > np.percentile(meanpopact[idx_T_still],100-perc)]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color=clrs_arealabels_low_high[ial,1],edgecolor='none')
    # ax.set_ylabel('FB (high)',fontsize=10)
    ax.set_xlabel('Population activity (deconv/dF0)',fontsize=10)
    ax.set_title(legendlabels[ial],fontsize=13)
    ax_nticks(ax,3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
# my_savefig(fig,savedir,'Hist_PopAct_FF_FB')
my_savefig(fig,savedir,'Hist_PopAct_FF_FB%s' % sessiondata['session_id'][ises])

#%%



#%% Correlate difference in activity metrics across sessions:

arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

perc                    = 25

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5

minnneurons             = 10
maxnoiselevel           = 20

#Regression output:
nboots                  = 100

#Correlation output:
corrdata_labdiff_ses          = np.full((nSessions),np.nan)
corrdata_targetarea_ses       = np.full((narealabelpairs,nSessions),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    diffdata            = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    targetareadata      = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    sessions[ises].celldata['nearby']
                                      ),axis=0))[0]
        
        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        if (len(idx_N1) < minnneurons) or (len(idx_N2) < minnneurons) or (len(idx_N3) < minnneurons):
            continue
        #Just mean activity:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        #Ratio:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        #Difference:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        targetact           = np.nanmean(respdata[idx_N3,:],axis=0)

        diffdata[ialp,:]     = meanpopact[idx_T_still]
        targetareadata[ialp,:] = targetact[idx_T_still]
        corrdata_targetarea_ses[ialp,ises] = pearsonr(diffdata[ialp,:],targetareadata[ialp,:])[0]
    if ~np.any(np.isnan(diffdata)):
        corrdata_labdiff_ses[ises] = pearsonr(diffdata[0,:],diffdata[1,:])[0]

print('Correlation across sessions: r = %1.2f (std = %1.2f), p = %1.2f' % (np.nanmean(corrdata_labdiff_ses),
                                                                            np.nanstd(corrdata_labdiff_ses),stats.ttest_1samp(corrdata_labdiff_ses,0,nan_policy='omit').pvalue))
for ialp in range(narealabelpairs):
    print('%s: r = %1.2f (std = %1.2f), p = %1.2f' % (arealabelpairs[ialp],np.nanmean(corrdata_targetarea_ses[ialp,:]),
                                                     np.nanstd(corrdata_targetarea_ses[ialp,:]),stats.ttest_1samp(corrdata_targetarea_ses[ialp,:],0,nan_policy='omit').pvalue))

#%% 
# ises = 0
# ialp = 0
# arealabelpairs  = [
#                     'V1lab-V1unl-PMunlL2/3',
#                     'PMlab-PMunl-V1unlL2/3',
#                     ]
# alp = arealabelpairs[ialp]
# idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
#                                             sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                             sessions[ises].celldata['nearby']
#                                             ),axis=0))[0]
# idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
#                                             sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                             sessions[ises].celldata['nearby']
#                                             ),axis=0))[0]

# meanpopact_N1          = np.nanmean(respdata[idx_source_N1,:],axis=0)
# meanpopact_N2          = np.nanmean(respdata[idx_source_N2,:],axis=0)
# vscale      = 0.015

# fig,axes    = plt.subplots(1,2,figsize=(6,3))


# # fig,axes = plt.subplots(2,2,figsize=(6,6))
# # fig,axes = plt.subplots(2,1,figsize=(3,6))
# ax=axes[0]
# ax.plot(meanpopact_N1,color='red',label=legendlabels[0],linewidth=0.1,alpha=1)
# ax.plot(meanpopact_N2,color='grey',label=legendlabels[1],linewidth=0.1,alpha=1)
# ax.set_xlabel('Trials')
# ax.set_ylabel('Population activity')
# ax.legend(['PMLab','PMUnl'],fontsize=9,frameon=False)
# ax = axes[1]
# # ax.scatter(meanpopact_N2,meanpopact_N1,color='k',s=0.1)
# ax.scatter(meanpopact_N2,meanpopact_N1,c=meanpopact_N1 - meanpopact_N2,s=0.2,cmap='coolwarm',vmin=-vscale,vmax=vscale)
# ax.text(0.1,0.8,'High FB',transform=ax.transAxes)
# ax.text(0.7,0.2,'Low FB',transform=ax.transAxes)
# ax.plot([-1,1],[-1,1],color='k',linewidth=0.2)
# ax.set_xlim(np.percentile(meanpopact_N1,[0,99.5]))
# ax.set_ylim(np.percentile(meanpopact_N1,[0,99.5]))
# ax.set_xlabel('PMUnl')
# ax.set_ylabel('PMLab')
# ax_nticks(ax, 3)
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=1)
# # my_savefig(fig,savedir,'Diff_LabUnl_GR%dsessions' % (nSessions), formats = ['png'])






#%% Show tuning curve when activity in the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
perc                    = 25

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5
alphathr                = 0.001 #threshold for correlation with cross area rate

minnneurons             = 10
maxnoiselevel           = 20
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells),np.nan)

#Regression output:
# nboots                  = 0
nboots                  = 250
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

#Correlation output:
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    # sessions[ises].celldata['nearby']
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    # sessions[ises].celldata['nearby']
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        # subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
        # idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
        # idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue
        
        #Just mean activity:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        #Ratio:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        #Difference:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        # meanpopact          = np.nanmean(respdata[idx_N2,:],axis=0) - np.nanmean(respdata[idx_N1,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        errorresp           = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            errorresp[:,i,0]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K1)))
            errorresp[:,i,1]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K2)))

        meanresp = meanresp - np.nanmin(meanresp[:,:,:],axis=(1,2),keepdims=True)

        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp[idx_N3]
        error_resp_split[ialp,:,:,idx_ses] = errorresp[idx_N3]

        regressdata          = np.full((N,3),np.nan)
        regress_sig          = np.full((N,2),0)
        for n in range(N):
            xdata = meanresp[n,:,0]
            ydata = meanresp[n,:,1]
            regressdata[n,:] = linregress(xdata,ydata)[:3]
        params_regress[idx_ses,ialp,:] = regressdata[idx_N3]

        if nboots:
            bootregressdata  = np.full((N,nboots,3),np.nan)
            bootregress_sig  = np.full((N,2),0)
            for iboot in range(nboots):
                meanrespboot            = np.empty([N,len(oris),2])
                for i,ori in enumerate(oris):
                    idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
                    idx_K1              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
                    idx_K2              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*perc//100,replace=False)
                    meanrespboot[:,i,0]     = np.nanmean(respdata[:,idx_K1],axis=1)
                    meanrespboot[:,i,1]     = np.nanmean(respdata[:,idx_K2],axis=1)
                for n in range(N):
                    bootregressdata[n,iboot,:] = linregress(meanrespboot[n,:,0],meanrespboot[n,:,1])[:3]

            bootregress_sig[regressdata[:,0]>np.percentile(bootregressdata[:,:,0],97.5,axis=1),0] = 1
            bootregress_sig[regressdata[:,0]<np.percentile(bootregressdata[:,:,0],2.5,axis=1),0] = -1
            bootregress_sig[regressdata[:,1]>np.percentile(bootregressdata[:,:,1],97.5,axis=1),1] = 1
            bootregress_sig[regressdata[:,1]<np.percentile(bootregressdata[:,:,1],2.5,axis=1),1] = -1

            sig_params_regress[idx_ses,ialp,:] = bootregress_sig[idx_N3]

        #Aligned:
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        mean_resp_split_aligned[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<alphathr) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

#%% Show some example neurons:

#%% use 
ialp = 0
ialp = 1
legendlabels        = ['FF','FB']

#%% Get good multiplicatively modulated cells:
#mutliplicative: 
idx_examples = np.all((params_regress[:,ialp,0]>np.nanpercentile(params_regress[:,ialp,0],90),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                        rangeresp>0.04,
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good divisive modulated cells:
idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                        rangeresp>0.04,
                       ),axis=0)

example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good additively modulated cells: 
#additive:
idx_examples = np.all((
                        # params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],70),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],75),
                        rangeresp>0.04,
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

idx_examples = np.all((
                       params_regress[:,ialp,0]>0.9,#slope within reasonable range of 1
                       params_regress[:,ialp,0]<1.1,   
                        # params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],80),
                        # params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                        rangeresp>0.04,
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good subtractive modulated cells: 
idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],70),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],25),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% 

#%% Plot two example neurons, one FF and one FB, with tuning curve and scatter side by side
example_cells = [
                    'LPE11086_2024_01_10_5_0048', #FF additive
                    'LPE11086_2024_01_10_2_0046', #FB Multiplicative
                    ]

#%% List of additional example FF cells:
example_cells = [
                    # 'LPE09665_2023_03_21_7_0011', #FF divisive
                    # 'LPE11086_2024_01_05_6_0103', #FF additive
                    # 'LPE09830_2023_04_10_5_0065', #FF additive
                    'LPE11086_2024_01_05_4_0002', #FF additive
                    'LPE11086_2024_01_05_5_0030', #FF additive
                    # 'LPE11086_2024_01_05_4_0235', #FF additive
                    # 'LPE11086_2024_01_05_4_0075', #FF additive
                    # 'LPE11086_2024_01_10_4_0017', #FF additive
                    # 'LPE11086_2024_01_10_5_0048', #FF additive
                    # 'LPE11086_2024_01_05_6_0304', #FF additive
                    'LPE11086_2024_01_05_4_0020', #FF additive
                    # 'LPE11086_2024_01_10_4_0055', #FF additive
                    # 'LPE11086_2024_01_10_4_0017', #FF additive
                    # 'LPE11086_2024_01_05_4_0040', #FF additive
                    'LPE11086_2024_01_05_5_0169', #FF multiplicative
                    # 'LPE10919_2023_11_06_0_0322', #FF subtractive/divisive
                    ]

#%%
example_cells = [
                    # 'LPE11086_2024_01_05_0_0030', #FB additive
                    'LPE12223_2024_06_10_1_0051', #FB multiplicative
                    # 'LPE11086_2024_01_10_3_0108', #FB multiplicative     
                    'LPE11086_2024_01_10_2_0046', #FB multiplicative
                    'LPE10885_2023_10_12_6_0014', #FB Multiplicative
                    'LPE10885_2023_10_12_5_0110', #FB Multiplicative
                    # 'LPE10885_2023_10_12_5_0036', #FB Multiplicative
                    # 'LPE10885_2023_10_12_4_0140', #FB Multiplicative
                    # 'LPE11086_2024_01_10_0_0009', #FB additive
                    # 'LPE10885_2023_10_23_1_0276', #FB divisive
                    # 'LPE10919_2023_11_06_5_0304', #FB divisive
                    # 'LPE11086_2024_01_10_0_0143', #FB additive
                ]

#%% Plot in two ways:
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

for example_cell in example_cells:
    idx_N = np.where(celldata['cell_id']==example_cell)[0][0]
    ialp = np.where(~np.isnan(mean_resp_split[:,0,0,idx_N]))[0][0]
    ustim = np.unique(sessions[ises].trialdata['Orientation'])
    x = mean_resp_split[ialp,:,0,idx_N]
    y = mean_resp_split[ialp,:,1,idx_N]
    xerror = error_resp_split[ialp,:,0,idx_N]
    yerror = error_resp_split[ialp,:,1,idx_N]
    
    # clrs_stimuli    = sns.color_palette('viridis',8)
    fig,axes = plt.subplots(1,2,figsize=(7*cm,3.5*cm))

    ax = axes[0]
    ax.scatter(ustim,x,color=clrs_arealabels_low_high[ialp,0],s=10)
    ax.plot(ustim,x,color=clrs_arealabels_low_high[ialp,0],linestyle='-',linewidth=0.5)
    # ax.errorbar(ustim,x,yerr=xerror,color='k',ls='None',linewidth=1)
    ax.errorbar(ustim,x,yerr=xerror,color=clrs_arealabels_low_high[ialp,0],ls='None',linewidth=1)

    ax.scatter(ustim,y,color=clrs_arealabels_low_high[ialp,1],s=10)
    ax.plot(ustim,y,color=clrs_arealabels_low_high[ialp,1],linestyle='-',linewidth=0.5)
    # ax.errorbar(ustim,y,yerr=yerror,color='k',ls='None',linewidth=1)
    ax.errorbar(ustim,y,yerr=yerror,color=clrs_arealabels_low_high[ialp,1],ls='None',linewidth=1)
    ax.set_xlabel('Orientation',fontsize=8,labelpad=padding)
    ax.set_ylabel('Response',fontsize=8,labelpad=padding)
    ax.set_xticks([0,90,180,270])
    ax.tick_params(axis='both', which='major', pad=padding)

    ax = axes[1]
    ax.scatter(x,y,color='k',s=6)
    ax.errorbar(x,y,xerr=xerror,yerr=yerror,color='k',ls='None',linewidth=1)
    b = linregress(x, y)
    xp = np.linspace(np.percentile(x,0),np.percentile(x,100)*1.1,100)
    ax.plot(xp,b[0]*xp+b[1],color=clrs_arealabelpairs[ialp],linestyle='-',linewidth=2)

    ax.text(0.5,0.05,'Slope: %1.2f\nOffest: %1.2f'%(b[0],b[1]),
                    transform=ax.transAxes,fontsize=6,color='k')
    ax.tick_params(axis='both', which='major', pad=padding)

    ax.plot([0,1],[0,1],color='grey',ls='--',linewidth=1)
    ax.set_xlim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
    ax.set_ylim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
    # ax.set_ylabel('%s high' % legendlabels[ialp],fontsize=8)
    # ax.set_xlabel('%s low' % legendlabels[ialp],fontsize=8)
    ax.set_ylabel('High',fontsize=8,labelpad=-3)
    ax.set_xlabel('Low',fontsize=8,labelpad=-3)
    ax_nticks(ax,2)
    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True, offset=1,trim=False)
    # my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell)
    # my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly','BaselineCorrected'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])

#%% Additional 4 example FF neurons:
example_cells = [
                    'LPE11086_2024_01_05_4_0002', #FF additive
                    'LPE11086_2024_01_05_5_0030', #FF additive
                    'LPE11086_2024_01_05_4_0020', #FF additive
                    'LPE11086_2024_01_05_5_0169', #FF multiplicative
                    ]
figtitle = 'LinearFit_additional_FF_example_neurons'

#%% Additional 4 example FB neurons:
example_cells = [
                    'LPE11086_2024_01_10_3_0108', #FB multiplicative     
                    'LPE10885_2023_10_12_4_0140', #FB Multiplicative
                    'LPE10919_2023_11_06_5_0304', #FB divisive
                    'LPE11086_2024_01_10_0_0143', #FB additive
                ]

figtitle = 'LinearFit_additional_FB_example_neurons'

#%% Plot some more example neurons only in high vs low format
fig,axes = plt.subplots(2,2,figsize=(4.1*cm,3.5*cm))

axes = axes.flatten()
for iexample_cell, example_cell in enumerate(example_cells[:4]):
    ax = axes[iexample_cell]

    idx_N = np.where(celldata['cell_id']==example_cell)[0][0]
    ialp = np.where(~np.isnan(mean_resp_split[:,0,0,idx_N]))[0][0]
    ustim = np.unique(sessions[ises].trialdata['Orientation'])
    x = mean_resp_split[ialp,:,0,idx_N]
    y = mean_resp_split[ialp,:,1,idx_N]
    xerror = error_resp_split[ialp,:,0,idx_N]
    yerror = error_resp_split[ialp,:,1,idx_N]

    ax.scatter(x,y,color='k',s=3)
    ax.errorbar(x,y,xerr=xerror,yerr=yerror,color='k',ls='None',linewidth=0.5)
    b = linregress(x, y)
    xp = np.linspace(np.percentile(x,0),np.percentile(x,100)*1.1,100)
    ax.plot(xp,b[0]*xp+b[1],color=clrs_arealabelpairs[ialp],linestyle='-',linewidth=1)

    ax.plot([0,1],[0,1],color='grey',ls='--',linewidth=0.5)
    ax.set_xlim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
    ax.set_ylim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
    # ax.set_ylabel('%s high' % legendlabels[ialp],fontsize=7)
    # ax.set_xlabel('%s low' % legendlabels[ialp],fontsize=7)
    ax_nticks(ax,2)
    ax.tick_params(axis='both', which='major', pad=1)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=2,trim=False)
my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly'),figtitle)
# my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell)
# my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly','BaselineCorrected'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])



#%% 
params_regress_mean = np.full((narealabelpairs,3),np.nan)
# clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    # idx_N =  np.array(celldata['gOSI']>0.5)
    # idx_N = params_regress[:,ialp,2] > 0.5
    # idx_N = corrsig_cells[ialp,:]==-1
    idx_N =  np.all((
                    # celldata['gOSI']>0.4,
                    rangeresp>minrangeresp,
                    # sig_params_regress[:,ialp,0]==1,
                    # corrsig_cells[ialp,:]==1,
                    # corrsig_cells[ialp,:]==1,
                    # np.any(mean_resp_split>0.5,axis=(0,1,2)),
                    # np.any(params_regress[:,:,2] > 0.5,axis=1),
                     ),axis=0)

    xdata = np.nanmean(mean_resp_split_aligned[ialp,:,0,idx_N].T,axis=1)
    ydata = np.nanmean(mean_resp_split_aligned[ialp,:,1,idx_N].T,axis=1)
    b = linregress(xdata,ydata)
    params_regress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = params_regress_mean[ialp,0]*xvals + params_regress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=1.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.7,s=35)
    ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
    ax.set_title(legendlabels[ialp],fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.text(0.1,0.8,'Slope: %1.2f\nOffest: %1.2f'%(params_regress_mean[ialp,0],params_regress_mean[ialp,1]),
            transform=ax.transAxes,fontsize=8)
    # ax.legend(frameon=False,loc='lower right')
    ax.set_xlabel('%s low (events/F0) '%(alp.split('-')[0]))
    ax.set_ylabel('%s high'%(alp.split('-')[0]))
    ax.set_xlim([0,np.nanmax([xdata,ydata])*1.1])
    ax.set_ylim([0,np.nanmax([xdata,ydata])*1.1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_%dsessions' % (nSessions), formats = ['png'])

#%% Fraction of significant multiplicative and additively modulated cells:
sign = 1
fig,axes = plt.subplots(1,1,figsize=(6*cm,4*cm))
ax = axes
idx_N =  rangeresp>minrangeresp
sigmat = np.empty((3,2))
countmat = np.empty((3,2))
for itype,(mult,add) in enumerate(zip([1,0,1],[0,1,1])):
    for ialp,alp in enumerate(arealabelpairs):
        Nsig = np.sum(np.all((
                    sig_params_regress[idx_N,ialp,0]==mult,
                    sig_params_regress[idx_N,ialp,1]==add,
                    corrsig_cells[ialp,idx_N]==sign,
                        ),axis=0))
        Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
        sigmat[itype,ialp] = Nsig
        countmat[itype,ialp] = Ntotal
        frac = Nsig/Ntotal
        xpos = itype*2 + ialp
        ax.bar(xpos,frac,width=0.8,color=clrs_arealabelpairs[ialp])
    pval = stats.chi2_contingency([[sigmat[itype,0], countmat[itype,0]-sigmat[itype,0]],
                            [sigmat[itype,1], countmat[itype,1]-sigmat[itype,1]]])[1]
    add_stat_annotation(ax,xpos-1,xpos+0,frac+0.01,pval,h=0,fontsize=9)
ax_nticks(ax,4)
ax.set_xticks(np.arange(3)*2+0.5,['Mult','Add','Both'])
ax.set_ylabel('Fraction of cells')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_affinemodulation_sig_posmod_%dsessions' % (nSessions))

#%% Fraction of significant multiplicative and additively modulated cells:
sign = -1
fig,axes = plt.subplots(1,1,figsize=(6*cm,4*cm))
ax = axes
idx_N =  rangeresp>minrangeresp
sigmat = np.empty((3,2))
countmat = np.empty((3,2))
for itype,(mult,add) in enumerate(zip([-1,0,-1],[0,-1,-1])):
    for ialp,alp in enumerate(arealabelpairs):
        Nsig = np.sum(np.all((
                    sig_params_regress[idx_N,ialp,0]==mult,
                    sig_params_regress[idx_N,ialp,1]==add,
                    corrsig_cells[ialp,idx_N]==sign,
                        ),axis=0))
        Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
        sigmat[itype,ialp] = Nsig
        countmat[itype,ialp] = Ntotal
        frac = Nsig/Ntotal
        xpos = itype*2 + ialp
        ax.bar(xpos,frac,width=0.8,color=clrs_arealabelpairs[ialp])
    pval = stats.chi2_contingency([[sigmat[itype,0], countmat[itype,0]-sigmat[itype,0]],
                            [sigmat[itype,1], countmat[itype,1]-sigmat[itype,1]]])[1]
    add_stat_annotation(ax,xpos-1,xpos+0,frac+0.01,pval,h=0,fontsize=9)
    ax_nticks(ax,3)
ax.set_xticks(np.arange(3)*2+0.5,['Mult','Add','Both'])
ax.set_ylabel('Fraction of cells')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_affinemodulation_sig_negmod_%dsessions' % (nSessions))


# #%% Fraction of significant multiplicative and additively modulated cells:
# fig,axes = plt.subplots(1,2,figsize=(6,3))
# for itype,(itype,itypelabel) in enumerate(zip([0,1],['mult','add'])):
#     ax = axes[itype]
#     for isign,(sign,signlabel) in enumerate(zip([1,-1],['pos','neg'])):
#         for ialp,alp in enumerate(arealabelpairs):
#             idx_N =  np.all((
#                 rangeresp>minrangeresp,
#                 corrsig_cells[ialp,:]==1,
#                      ),axis=0)
#             # frac = np.sum(sig_params_regress[:,ialp,itype]==sign) / np.sum(~np.isnan(sig_params_regress[:,ialp,itype]))
#             Nsig = np.sum(sig_params_regress[idx_N,ialp,itype]==sign)
#             Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,itype]))
#             frac = Nsig/Ntotal
#             xpos = isign*2 + ialp
#             ax.bar(xpos,frac,width=0.8,color=clrs_arealabelpairs[ialp])
#             ax.text(xpos,.01,'%d/\n%d' % (Nsig,Ntotal),
#                 ha='center',va='bottom',fontsize=7)
#             ax.set_xticks(range(4))
#             ax.set_xticks([.5,2.5],['+','-'],fontsize=15)
#             ax.set_title(itypelabel)
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'FF_FB_affinemodulation_sig_mod_labunldiff_%dsessions' % (nSessions), formats = ['png'])

#%%
fracmat     = np.full((3,3,narealabelpairs+1),np.nan)
nsigmat     = np.full((3,3,narealabelpairs),np.nan)
ntotalmat   = np.full((3,3,narealabelpairs),np.nan)
testmat     = np.full((3,3),np.nan)
ncomparisons = 9
for ialp,alp in enumerate(arealabelpairs):
    # for imult, mult in enumerate([-1,0,1]):
    for imult, mult in enumerate([1,0,-1]):
        for iadd, add in enumerate([-1,0,1]):
            idx_N =  np.all((
                rangeresp>0.05,
                     ),axis=0)
            Nsig = np.sum(np.all((
                                sig_params_regress[idx_N,ialp,0]==mult,
                                sig_params_regress[idx_N,ialp,1]==add,
                                ),axis=0))
            Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
            frac = (Nsig/Ntotal) * 100
            nsigmat[imult,iadd,ialp] = Nsig
            ntotalmat[imult,iadd,ialp] = Ntotal
            fracmat[imult,iadd,ialp] = frac
fracmat[:,:,2] = fracmat[:,:,1] - fracmat[:,:,0]

for imult, mult in enumerate([1,0,-1]):
    for iadd, add in enumerate([-1,0,1]):
        data = np.array([[nsigmat[imult,iadd,0], ntotalmat[imult,iadd,0]-nsigmat[imult,iadd,0]],
                         [nsigmat[imult,iadd,1], ntotalmat[imult,iadd,1]-nsigmat[imult,iadd,1]]])
        testmat[imult,iadd] = stats.chi2_contingency(data)[1]  # p-value
testmat = testmat * ncomparisons  #bonferroni correction

fig,axes = plt.subplots(1,3,figsize=(9,3))
for ialp in range(narealabelpairs+1):
    ax = axes[ialp]
    if ialp < narealabelpairs:
        vmin,vmax = 0,25
        # cmap = 'Purples'
        cmap = 'viridis'
        # cmap = 'magma'
        # cmap = 'Greens'
    else:
        vmin,vmax = -5,5
        # cmap = 'bwr'
        cmap = 'PiYG'
    im = ax.imshow(fracmat[:,:,ialp],vmin=vmin,vmax=vmax,cmap=cmap)

    ax.set_xticks([0,1,2],['Sub','None','Add'])
    # ax.set_yticks([0,1,2],['Div','None','Mult'])
    ax.set_yticks([0,1,2],['Mult','None','Div'])
    ax.set_xlabel('Addition')
    if ialp == 0:
        ax.set_ylabel('Multiplicative')
    ax.set_title(legendlabels[ialp] if ialp < narealabelpairs else 'Diff (FB-FF)')
    for i in range(3):
        for j in range(3):
            if ialp != narealabelpairs:
                # ax.text(j,i,'%1.2f' % fracmat[i,j,ialp],ha='center',va='center',color='white' if fracmat[i,j,ialp]<20 else 'black')
                ax.text(j,i,'%2.1f%%' % fracmat[i,j,ialp],ha='center',va='center',color='white' if fracmat[i,j,ialp]<20 else 'black')
            else: 
                # ax.text(j,i,'%s%2.1f%%\n%s' % ('+' if fracmat[i,j,ialp]>0 else '',fracmat[i,j,ialp],get_sig_asterisks(testmat[i,j])),ha='center',va='center',color='white' if fracmat[i,j,ialp]>0.2 else 'black')
                ax.text(j,i,'%s%2.1f%%\n%s' % ('+' if fracmat[i,j,ialp]>0 else '',fracmat[i,j,ialp],get_sig_asterisks(testmat[i,j])),
                        # ha='center',va='center',color='white' if fracmat[i,j,ialp]>0.2 else 'black')
                        ha='center',va='center',color='black')
                # ax.text(j,i,'%+2.1f%%' % ('+' if fracmat[i,j,ialp]>0 else '') + '%2.1f%%' % fracmat[i,j,ialp],ha='center',va='center',color='white' if fracmat[i,j,ialp]>0.2 else 'black')

    fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04,label='% sign. cells')
plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Affine_sig_mod_FF_FB_heatmap_%dsessions' % (nSessions))


#%% 
legendlabels = ['FF','FB']
fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharey=True)
for iparam in range(2):
    ax = axes[iparam]
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        bins = np.arange(-0.25,5,0.015)
        xlims = [0,3]
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        bins = np.arange(-0.05,0.08,0.0001)
        xlims = [-0.01,0.05]
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    handles = []
    for ialp,alp in enumerate(arealabelpairs):
        
        idx_N =  rangeresp>minrangeresp
        # idx_N =  sig_params_regress[:,ialp,1]==1
        idx_N = np.all((
                rangeresp>minrangeresp,
                np.any(corrsig_cells==1,axis=0),
                # np.any(corrsig_cells==-1,axis=0),
                ),axis=0)
        sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],
                     alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
        handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.95,markersize=6,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
        ncells = np.sum(~np.isnan(params_regress[idx_N,ialp,iparam]))

        ax.text(0.7, 0.1+ialp*0.1, 'n=%d' % ncells, 
                transform=ax.transAxes,fontsize=6,color=clrs_arealabelpairs[ialp])
        
     # ax.axvline(np.nanmean(params_regress[idx_N,ialp,iparam]),color=clrs_arealabelpairs[ialp],ls='--',linewidth=1)
        # ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    h,p = stats.ttest_ind(params_regress[idx_N,0,iparam],
                            params_regress[idx_N,1,iparam],nan_policy='omit')
    p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
    # ax.text(0.6, 0.15, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes,fontsize=9)
    ax.text(0.45, 0.5, '%s' % (get_sig_asterisks(p,return_ns=True)),
            transform=ax.transAxes,fontsize=10)
    ax.set_yticks([0,0.5,1.0])
    ax.set_xlim(xlims)
    ax.set_ylabel('Cumulative fraction of cells')

    # ax.legend(handles,legendlabels,fontsize=9,frameon=False,loc='center right')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=2)

my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_RangeResp_posmod_cumhistcoefs_%dGRsessions' % (nSessions))
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_RangeResp_negmod_cumhistcoefs_%dGRsessions' % (nSessions))
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])

# %%
celldata['meanact']     = np.nanmean(mean_resp_split,axis=(0,1,2))
celldata['slope']       = np.nanmean(params_regress[:,:,0],axis=1)
celldata['offset']      = np.nanmean(params_regress[:,:,1],axis=1)
celldata['affine_R2']   = np.nanmean(params_regress[:,:,2],axis=1)
celldata['rangeresp']   = rangeresp

#%%
fig,axes = plt.subplots(2,3,figsize=(9,6))
for ivarx,varx in enumerate(['slope','offset','affine_R2']):
    for ivary,vary in enumerate(['gOSI','rangeresp']):
        ax = axes[ivary,ivarx]
        ax.scatter(celldata[vary],celldata[varx],s=3,alpha=0.2,color='grey')
        ax.set_xlabel(vary)
        ax.set_ylabel(varx)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

#%%


# def plot_binned_ci(ax,xdata,ydata,bins,clr):
#     bincenters      = (bins[:-1]+bins[1:])/2 #get bin centers

#     idx_notnan = np.logical_and(~np.isnan(xdata),~np.isnan(ydata))
#     xdata = xdata[idx_notnan]
#     ydata = ydata[idx_notnan]
#     ymeandata = binned_statistic(xdata, ydata, statistic='mean', 
#                             bins=bins)[0]

#     nboots = 250
#     for iboot in range(nboots):
#         idx = np.random.choice(len(xdata),size=len(xdata),replace=True)
#         xboot = xdata[idx]
#         yboot = ydata[idx]
#         bootdata[ialp,:,iboot] = binned_statistic(xboot, yboot, statistic='mean', 
#                             bins=bins)[0]
#     bootci[ialp,:,:] = np.percentile(bootdata[ialp,:,:],[(100-ci)/2,100-(100-ci)/2],axis=1)
#     ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],marker='o',linestyle='None',markersize=4)
#     handles.append(ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],
#         linewidth=1.5)[0])
#     ax.fill_between(bincenters,bootci[ialp,0,:],bootci[ialp,1,:],color=clrs_arealabelpairs[ialp],
#                     alpha=0.3)




#%% Is the effect similar for the two areas, but 
# just dependent on a difference in activity levels?
# Is modulation more multiplicative for larger activity levels, stimuli with 

ci              = 95 #bootstrapped confidence interval
nboots          = 250 #number of bootstrap samples
percspacing     = 2.5 #bins chosen to have approx equal number of points
percentiles     = np.arange(0,100+percspacing,percspacing)
percentiles[percentiles==100] = 99.75 #avoid issues with max value
bins            = np.nanpercentile(mean_resp_split,percentiles)
bins            = bins[bins>0] #remove duplicate bins at 0
bincenters      = (bins[:-1]+bins[1:])/2 #get bin centers

resp_mod = mean_resp_split[:,:,1,:] - mean_resp_split[:,:,0,:]

fig,axes = plt.subplots(1,3,figsize=(9,3),sharex=True,sharey=False)
ax = axes[0]
handles = []
bootdata = np.full((narealabelpairs,len(bins)-1,nboots),np.nan)
bootci = np.full((narealabelpairs,2,len(bins)-1),np.nan)
for ialp,alp in enumerate(arealabelpairs):
    idx_N =  rangeresp>minrangeresp
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    ydata = resp_mod[np.ix_([ialp],range(16),idx_N)].flatten()
    # plot_binned_ci(ax,xdata,ydata,bins,clrs_arealabelpairs[ialp])
    
    idx_notnan = np.logical_and(~np.isnan(xdata),~np.isnan(ydata))
    xdata = xdata[idx_notnan]
    ydata = ydata[idx_notnan]
    ymeandata = binned_statistic(xdata, ydata, statistic='mean', 
                            bins=bins)[0]

    for iboot in range(nboots):
        idx = np.random.choice(len(xdata),size=len(xdata),replace=True)
        xboot = xdata[idx]
        yboot = ydata[idx]
        bootdata[ialp,:,iboot] = binned_statistic(xboot, yboot, statistic='mean', 
                            bins=bins)[0]
    bootci[ialp,:,:] = np.percentile(bootdata[ialp,:,:],[(100-ci)/2,100-(100-ci)/2],axis=1)
    ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],marker='o',linestyle='None',markersize=4)
    handles.append(ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],
        linewidth=1.5)[0])
    ax.fill_between(bincenters,bootci[ialp,0,:],bootci[ialp,1,:],color=clrs_arealabelpairs[ialp],
                    alpha=0.3)
# ax.legend(handles,legendlabels,fontsize=11,frameon=False,loc='best')
ax.set_ylabel('Modulation')
ax.axhline(0,color='grey',ls='--',linewidth=1)
ax.set_xlim([0,bincenters[-1]*1.01])
ax_nticks(ax,3)
for ibin in range(len(bincenters)):
    if bootci[0,0,ibin] > bootci[1,1,ibin] or bootci[0,1,ibin] < bootci[1,0,ibin]:
        # ax.plot(bincenters[ibin],-0.001,'k*',markersize=6)
        ax.plot(bincenters[ibin],0.01,'k*',markersize=6)

ax = axes[1]
for ialp,alp in enumerate(arealabelpairs):
    idx_N = np.all((
                rangeresp>minrangeresp,
                # corrsig_cells[ialp,:]==1,
                sig_params_regress[:,ialp,0]==1,
                ),axis=0)
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    ydata = resp_mod[np.ix_([ialp],range(16),idx_N)].flatten()
    ymeandata = binned_statistic(xdata, ydata, statistic='mean',bins=bins)[0]
    ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],linewidth=2)
    
    idx_notnan = np.logical_and(~np.isnan(xdata),~np.isnan(ydata))
    xdata = xdata[idx_notnan]
    ydata = ydata[idx_notnan]
    ymeandata = binned_statistic(xdata, ydata, statistic='mean', 
                            bins=bins)[0]

    for iboot in range(nboots):
        idx = np.random.choice(len(xdata),size=len(xdata),replace=True)
        xboot = xdata[idx]
        yboot = ydata[idx]
        bootdata[ialp,:,iboot] = binned_statistic(xboot, yboot, statistic='mean', 
                            bins=bins)[0]
    bootci[ialp,:,:] = np.percentile(bootdata[ialp,:,:],[(100-ci)/2,100-(100-ci)/2],axis=1)
    ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],marker='o',linestyle='None',markersize=4)
    handles.append(ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],
        linewidth=1.5)[0])
    ax.fill_between(bincenters,bootci[ialp,0,:],bootci[ialp,1,:],color=clrs_arealabelpairs[ialp],
                    alpha=0.3)
ax.legend(handles,legendlabels,fontsize=11,frameon=False,loc='best')
ax.set_xlim([0,bincenters[-1]])
ax.set_ylim([0,.1])
ax.set_xlabel('Activity')
ax_nticks(ax,3)
for ibin in range(len(bincenters)):
    if bootci[0,0,ibin] > bootci[1,1,ibin] or bootci[0,1,ibin] < bootci[1,0,ibin]:
        # ax.plot(bincenters[ibin],-0.001,'k*',markersize=6)
        ax.plot(bincenters[ibin],0.01,'k*',markersize=6)

ax = axes[2]
for ialp,alp in enumerate(arealabelpairs):
    idx_N = np.all((          
                rangeresp>minrangeresp,
                # corrsig_cells[ialp,:]!=-1,
                sig_params_regress[:,ialp,1]==1,
                ),axis=0)
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    ydata = resp_mod[np.ix_([ialp],range(16),idx_N)].flatten()
    ymeandata = binned_statistic(xdata, ydata, statistic='mean',bins=bins)[0]
    ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],
            linewidth=2)
    
    idx_notnan = np.logical_and(~np.isnan(xdata),~np.isnan(ydata))
    xdata = xdata[idx_notnan]
    ydata = ydata[idx_notnan]
    ymeandata = binned_statistic(xdata, ydata, statistic='mean', 
                            bins=bins)[0]

    for iboot in range(nboots):
        idx = np.random.choice(len(xdata),size=len(xdata),replace=True)
        xboot = xdata[idx]
        yboot = ydata[idx]
        bootdata[ialp,:,iboot] = binned_statistic(xboot, yboot, statistic='mean', 
                            bins=bins)[0]
    bootci[ialp,:,:] = np.percentile(bootdata[ialp,:,:],[(100-ci)/2,100-(100-ci)/2],axis=1)
    ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],marker='o',linestyle='None',markersize=4)
    handles.append(ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],
        linewidth=1.5)[0])
    ax.fill_between(bincenters,bootci[ialp,0,:],bootci[ialp,1,:],color=clrs_arealabelpairs[ialp],
                    alpha=0.3)
ax.set_xlim([0,bincenters[-1]])
ax_nticks(ax,3)
ax.set_ylim([0,.1])
ax_nticks(ax,3)
for ibin in range(len(bincenters)):
    if bootci[0,0,ibin] > bootci[1,1,ibin] or bootci[0,1,ibin] < bootci[1,0,ibin]:
        # ax.plot(bincenters[ibin],-0.001,'k*',markersize=6)
        ax.plot(bincenters[ibin],0.01,'k*',markersize=6)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'FF_FB_Modulation_vs_Activity_%dGRsessions' % (nSessions))


#%% Is the effect similar for the two areas, but 
# just dependent on a difference in activity levels?
# Is modulation more multiplicative for larger activity levels, stimuli with 
mincounts = 20

vartocontrol = 'gOSI'
# vartocontrol = 'rangeresp'
# vartocontrol = 'meanact'
bins = np.linspace(celldata[vartocontrol].min(),np.nanpercentile(celldata[vartocontrol],99),20)
fig,axes = plt.subplots(2,2,figsize=(5,5),sharex='row',sharey='row')
for ivar,var in enumerate(['slope','offset']):
    for ialp,alp in enumerate(arealabelpairs):
        ax = axes[ivar,0]

        idx_N = celldata['arealayerlabel'] == alp.split('-')[2]
        xdata = celldata[vartocontrol][idx_N]
        ydata = celldata[var][idx_N]

        idx_notnan = np.logical_and(~np.isnan(xdata),~np.isnan(ydata))
        xdata = xdata[idx_notnan]
        ydata = ydata[idx_notnan]

        ax.set_xlim(np.percentile(bins,[0,100]))
        ax.scatter(xdata,ydata,s=3,alpha=0.2,color=clrs_arealabelpairs[ialp])
        ax.set_xlabel(vartocontrol)
        ax.set_ylabel(var)

        ax = axes[ivar,1]
        ymeandata,bin_edges = binned_statistic(xdata, ydata, statistic='mean', 
                                bins=bins)[:2]
        bincounts = np.histogram(xdata,bins=bin_edges)[0]
        ymeandata[bincounts<mincounts] = np.nan
        ax.plot(bin_edges[:-1],ymeandata,color=clrs_arealabelpairs[ialp],
                linewidth=2)
        ax.set_ylim(np.nanpercentile(ydata,[1,99]))
        ax.set_xlim(np.nanpercentile(bins,[0,100]))
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Control_AffineCoefs_%s_%dGRsessions' % (vartocontrol,nSessions), formats = ['png'])

#%% Is there a difference in the distribution of activity across the stimuli across the areas:
fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=False,sharey=True)
resp_mod = mean_resp_split[:,:,1,:] - mean_resp_split[:,:,0,:]

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[0]
    # xdata = mean_resp_split[ialp,:,0,:].flatten()
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()

    ydata = resp_mod[ialp].flatten()

    # ax.scatter(xdata,ydata,s=2,alpha=0.3,color=clrs_arealabelpairs[ialp])
    # ymeandata,bin_edges = binned_statistic(xdata, ydata, statistic='mean', 
                            # bins=np.arange(0,0.8,0.025))[:2]
    # ax.plot(bin_edges[:-1],ymeandata,color=clrs_arealabelpairs[ialp],
    #         linewidth=2)
    ax.set_ylim([0,1])
    ax.set_xlabel('Activity')

    sns.histplot(xdata,bins=np.arange(0,0.1,0.0005),fill=False,stat='probability',
                 cumulative=True,
                 color=clrs_arealabelpairs[ialp],element='step',ax=ax)
    ax = axes[1]
    sns.histplot(ydata,bins=np.arange(-0.1,0.1,0.0005),fill=False,stat='probability',
                 cumulative=True,
                 color=clrs_arealabelpairs[ialp],element='step',ax=ax)
    ax.set_ylim([0,1])
    ax.set_xlabel('Modulation')

plt.tight_layout()
# my_savefig(fig,savedir,'Control_Activity_Diff_vs_Mod_%dGRsessions' % (nSessions), formats = ['png'])

#%% Show for positive and negatively correlated neurons only:
params_regress_mean = np.full((narealabelpairs,3),np.nan)

legendlabels = ['FF','FB']
# clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(2,3,figsize=(7,4))
for isign,sign in enumerate([-1,1]):
    print(sign)
    for iparam in range(2):
        ax = axes[isign,iparam]
        if iparam == 0:
            ax.set_xlabel('Multiplicative Slope')
            bins = np.arange(-0.5,3,0.05)
            ax.axvline(1,color='grey',ls='--',linewidth=1)
        else:
            ax.set_xlabel('Additive Offset')
            bins = np.arange(-0.025,0.05,0.0015)
            ax.axvline(0,color='grey',ls='--',linewidth=1)
        handles = []
        for ialp,alp in enumerate(arealabelpairs):
            ax = axes[isign,iparam]
            # idx_N = params_regress[:,ialp,2] > 0.5
            idx_N =  np.all((
                             celldata['gOSI']>0.4,
                            #  rangeresp>0.04,
                             corrsig_cells[ialp,:]==sign),axis=0)
            # print(np.sum(idx_N))
            sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                        color=clrs_arealabelpairs[ialp],
                        alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
            handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
                    color=clrs_arealabelpairs[ialp],marker='v')[0])
            ax.legend(handles,legendlabels,fontsize=9,frameon=False)
        
            ax = axes[isign,2]
            xdata = np.nanmean(mean_resp_split_aligned[ialp,:,0,idx_N].T,axis=1)
            ydata = np.nanmean(mean_resp_split_aligned[ialp,:,1,idx_N].T,axis=1)
            b = linregress(xdata,ydata)
            params_regress_mean[ialp,:] = b[:3]
            xvals = np.arange(0,3,0.1)
            yvals = params_regress_mean[ialp,0]*xvals + params_regress_mean[ialp,1]
            ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=1.3)
            ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.7,s=25)
            ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
            # ax.set_title(,fontsize=12,color=clrs_arealabelpairs[ialp])
            ax.text(0.6,0.15*ialp,'Slope: %1.2f\nOffest: %1.2f'%(params_regress_mean[ialp,0],params_regress_mean[ialp,1]),
                    transform=ax.transAxes,fontsize=8,color=clrs_arealabelpairs[ialp])
            ax.set_xlabel('%s low (events/F0) '%(alp.split('-')[0]))
            ax.set_ylabel('%s high'%(alp.split('-')[0]))
            ax.set_xlim([0,np.nanmax([xdata,ydata])*1.1])
            ax.set_ylim([0,np.nanmax([xdata,ydata])*1.1])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_PosNeg_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])


#%% Show correlation of slope and offset
fig,axes = plt.subplots(1,2,figsize=(5,2.5),sharey=True,sharex=True)
for ialp,alp in enumerate(legendlabels): #Mult and Add
    ax = axes[ialp]
    idx_N = np.all((
            celldata['gOSI']>0,
            rangeresp>0.04,
                ),axis=0)
    x = params_regress[idx_N,ialp,0]
    y = params_regress[idx_N,ialp,1]
    # c = celldata['gOSI'][idx_N][~np.isnan(y)]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    sns.scatterplot(x=x,y=y,s=5,alpha=0.2,color=clrs_arealabelpairs[ialp],ax=ax)
    # sns.scatterplot(x=x,y=y,s=5,alpha=0.2,c=celldata['gOSI'][idx_N],ax=ax)
    # ax.scatter(x=x,y=y,s=5,alpha=0.2,c=c)
    
    ax.set_xlim(np.nanpercentile(params_regress[idx_N,:,0],[0.2,99.8]))
    ax.set_ylim(np.nanpercentile(params_regress[idx_N,:,1],[0.2,99.8]))
    ax_nticks(ax, 3)
    ax.set_xlabel(u'Slope')
    ax.set_ylabel(u'Offset')
    ax.set_title(legendlabels[ialp],fontsize=11)
    ax.text(0.5,0.8,'r = %.2f' % (stats.pearsonr(x,y)[0]),transform=ax.transAxes,fontsize=9) #print(stats.pearsonr(x,y)[0])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Corr_Mult_Add_Mean_FF_FB_MinRespSubtracted_GR%dsessions' % (nSessions))
# my_savefig(fig,savedir,'Corr_Mult_Add_Mean_FF_FB_BaselineSubtracted_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'Corr_Mult_Add_Mean_FF_FB_GR%dsessions' % (nSessions), formats = ['png'])


#%% Check whether epochs of endogenous high feedforward activity are associated with a specific modulation of the 
# tuning curve of PM neurons and vice versa for feedback. Because the population activity in V1 and PM cofluctuates,
#  just taking the level of V1 or PM activity would confound the analysis with local activity levels. 
# So therefore I took the 10% of trials with the labeled cells being more active than unlabeled cells vs the 10% trials 
# with unlabeled cells being more active than labeled cells (e.g. for FF: mean of V1lab - mean of V1unl). This would 
# be a proxy of epochs of particularly high FF activity, vs epochs of low FF activity (while controlling for overall 
# activity levels). Then the population tuning curve of PMunl or PMlab is plotted computed on these trials separately.
# You can see that high FF activity has very small divisive effect, while high FB activity has a clear multiplicative 
# effect. I also checked the effect on individual neurons (fitting affine modulation per neuron) but they mainly reflect 
# the mean. There are also additive effects, but the magnitude of the additive effects does not seem larger for PM cells 
# when FF ratio is high (edited) 


arealabelpairs  = [
                    'V1labL2/3-V1unlL2/3-PMunlL2/3',
                    'V1labL2/3-V1unlL2/3-PMunlL5',
                    'PMlabL2/3-PMunlL2/3-V1unlL2/3',
                    'PMlabL5-PMunlL5-V1unlL2/3',
                    ]

arealabelpairs  = [
                    'V1labL2/3-V1unlL2/3-PMunlL2/3',
                    'V1labL2/3-V1unlL2/3-PMlabL2/3',
                    'V1labL2/3-V1unlL2/3-PMunlL5',
                    'V1labL2/3-V1unlL2/3-PMlabL5',
                    'PMlabL2/3-PMunlL2/3-V1unlL2/3',
                    'PMlabL2/3-PMunlL2/3-V1labL2/3',
                    'PMlabL5-PMunlL5-V1unlL2/3',
                    'PMlabL5-PMunlL5-V1labL2/3',
                    ]

arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'V1lab-V1unl-PMlabL2/3',
                    # 'V1lab-V1unl-PMunlL5',
                    # 'V1lab-V1unl-PMlabL5',
                    'PMlab-PMunl-V1unlL2/3',
                    'PMlab-PMunl-V1labL2/3',
                    ]

# arealabelpairs  = [
                    # 'V1unl-PMunl-PMunl',
                    # 'V1unl-PMunl-PMlab',
                    # 'V1lab-PMunl-PMunl',
                    # 'V1lab-PMunl-PMlab',
                    # 'PMunl-V1unl-V1unl',
                    # 'PMunl-V1unl-V1lab',
                    # 'PMlab-V1unl-V1unl',
                    # 'PMlab-V1unl-V1lab',
                    # ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())

nboots                  = 100
perc                    = 25
minnneurons             = 10
maxnoiselevel           = 20

alphathr                = 0.001
# maxnoiselevel           = 100
# mineventrate            = 0

mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
corrdata                = np.full((narealabelpairs,nSessions),np.nan)
corrdata_boot           = np.full((narealabelpairs,nSessions,nboots),np.nan)
corrdata_confounds      = np.full((narealabelpairs,nSessions,3),np.nan)
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)

# valuematching           = 'pop_coupling'
valuematching           = None
nmatchbins              = 5

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    # respdata            = zscore(sessions[ises].respmat, axis=1)
    # respdata            = zscore(sessions[ises].respmat_behavout, axis=1)

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # respdata            = sessions[ises].respmat_behavout / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    # idx_nearby          = filter_nearlabeled(sessions[ises],radius=50)

    for ialp,alp in enumerate(arealabelpairs):
        # idx_N1              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0]
        # idx_N2              = sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1]
        # idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2]

        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    # sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                    #   sessions[ises].celldata['OSI']>0.5,
                                    #   sessions[ises].celldata['gOSI']>0.5,
                                    # idx_nearby,

                                      ),axis=0))[0]
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    # sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                        # sessions[ises].celldata['gOSI']>0.5,
                                        #   sessions[ises].celldata['OSI']>0.5,
                                    # idx_nearby,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                    #   sessions[ises].celldata['tuning_var']>0.025,
                                    #   sessions[ises].celldata['gOSI']>np.nanpercentile(sessions[ises].celldata['gOSI'],50),
                                    #   sessions[ises].celldata['OSI']>0.5,
                                    #   sessions[ises].celldata['gOSI']>0.5,
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        # if len(idx_N1) < minnneurons or len(idx_N2) < minnneurons or len(idx_N3) < minnneurons:
            # continue

        # if len(idx_N1) < minnneurons:
        #     continue

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue

        if valuematching is not None:
            #Get value to match from celldata for V1 matching
            values      = sessions[ises].celldata[valuematching].to_numpy()
            idx_joint   = np.concatenate((idx_N1,idx_N2))
            group       = np.concatenate((np.zeros(len(idx_N1)),np.ones(len(idx_N2))))
            idx_sub     = value_matching(idx_joint,group,values[idx_joint],bins=nmatchbins,showFig=False)
            idx_N1      = np.intersect1d(idx_N1,idx_sub) #recover subset from idx_joint
            idx_N2      = np.intersect1d(idx_N2,idx_sub)
        
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        # meanpopact          = np.nanmean(zscore(respdata[idx_N1,:],axis=1),axis=0)

        corrdata_confounds[ialp,ises,0]      = np.corrcoef(meanpopact,sessions[ises].respmat_runspeed)[0,1]
        corrdata_confounds[ialp,ises,1]      = np.corrcoef(meanpopact,sessions[ises].respmat_videome)[0,1]
        corrdata_confounds[ialp,ises,2]      = np.corrcoef(meanpopact,sessions[ises].respmat_pupilarea)[0,1]

        corrdata[ialp,ises]      = np.corrcoef(meanpopact,np.nanmean(respdata[idx_N3,:],axis=0))[0,1]

        # sampleNneurons = min(np.sum(idx_N1),np.sum(idx_N2),np.sum(idx_N3))
        sampleNneurons = min(len(idx_N1),len(idx_N2),len(idx_N3))

        for iboot in range(nboots):
            bootidx_N1          = np.random.choice(idx_N1,sampleNneurons,replace=True)
            bootidx_N2          = np.random.choice(idx_N2,sampleNneurons,replace=True)
            bootidx_N3          = np.random.choice(idx_N3,sampleNneurons,replace=True)
            # corrdata_boot[ialp,ises,iboot] = np.corrcoef(np.nanmean(respdata[bootidx_N1,:],axis=0) - np.nanmean(respdata[bootidx_N2,:],axis=0),
                                            #   np.nanmean(respdata[bootidx_N3,:],axis=0))[0,1]
            corrdata_boot[ialp,ises,iboot] = np.corrcoef(np.nanmean(respdata[bootidx_N1,:],axis=0),
                                              np.nanmean(respdata[bootidx_N3,:],axis=0))[0,1]
        
        # idx_K1              = meanpopact < np.nanpercentile(meanpopact,perc)
        # idx_K2              = meanpopact > np.nanpercentile(meanpopact,100-perc)
        # # compute meanresp for trials with low and high difference in lab-unl activation
        # meanresp            = np.empty([N,len(oris),2])
        # ori_ses             = sessions[ises].trialdata['Orientation']
        # oris                = np.unique(ori_ses)
        # for i,ori in enumerate(oris):
        #     meanresp[:,i,0] = np.nanmean(respdata[:,np.logical_and(ori_ses==ori,idx_K1)],axis=1)
        #     meanresp[:,i,1] = np.nanmean(respdata[:,np.logical_and(ori_ses==ori,idx_K2)],axis=1)
        
        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            idx_T               = ori_ses == ori
            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0] = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1] = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response during still trials
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        # meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,0] = meanresp_pref[:,:,0] / tempmax
        # meanresp_pref[:,:,1] = meanresp_pref[:,:,1] / tempmax

        # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<alphathr) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# Fit gain coefficient for each neuron and compare labeled and unlabeled neurons:
N = len(celldata)
params_regress = np.full((N,narealabelpairs,3),np.nan)
for iN in tqdm(range(N),total=N,desc='Fitting gain for each neuron'):
    for ialp,alp in enumerate(arealabelpairs):
        xdata = mean_resp_split[ialp,:,0,iN]
        ydata = mean_resp_split[ialp,:,1,iN]
        params_regress[iN,ialp,:] = linregress(xdata,ydata)[:3]

#%%
# idx_sigN = corrsig_cells[0,:]==1
# idx_sigN = corrsig_cells[0,:]==-1
# plt.hist(corrdata_cells[0,idx_sigN].flatten())
# corrsig_cells[ialp,idx_ses]==-1


#%%
plotdata = np.nanmean(corrdata_boot,axis=2)

fig,ax = plt.subplots(1,1,figsize=(2.5,3))
df = pd.DataFrame({'correlation': plotdata.flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
sns.despine(fig=fig, top=True, right=True,offset=3)
ax.set_xticklabels(arealabelpairs,rotation=90,fontsize=8)

# my_savefig(fig,savedir,'FF_FB_poprate_arealayerpairs_GR_%dsessions' % (nSessions))

#%% 

fig,axes = plt.subplots(1,3,figsize=(9,3))
ax=axes[0]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,0].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Running speed')

ax=axes[1]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,1].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Video ME')

ax=axes[2]
df = pd.DataFrame({'correlation': corrdata_confounds[:,:,2].flatten(),'arealabelpair': np.repeat(arealabelpairs,nSessions)})
sns.pointplot(data=df,x='arealabelpair',y='correlation',estimator=np.nanmean,errorbar=('ci', 90),palette='pastel',ax=ax)
sns.stripplot(data=df, x="arealabelpair", y="correlation", palette='pastel',dodge=False, alpha=1, legend=False,ax=ax)
ax.axhline(0,linestyle='--',color='k')
ax.set_title('Pupil size')

for ax in axes:
    ax.set_ylim([-0.3,0.3])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
axes[0].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)
axes[1].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)
axes[2].set_xticklabels(arealabelpairs,rotation=90,fontsize=8)

my_savefig(fig,savedir,'FF_FB_poprate_confounds_GR_%dsessions' % (nSessions))

#%% 
params_regress_mean = np.full((narealabelpairs,3),np.nan)
# clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    # idx_N =  np.array(celldata['gOSI']>0.5)
    # idx_N = params_regress[:,ialp,2] > 0.5
    # idx_N = corrsig_cells[ialp,:]==-1

    idx_N =  np.all((
                    # celldata['gOSI']>0.5,
                    # celldata['nearby'],
                    # corrsig_cells[ialp,:]==1,
                    # np.any(mean_resp_split>0.5,axis=(0,1,2)),
                    np.any(params_regress[:,:,2] > 0.5,axis=1),
                     ),axis=0)

    xdata = np.nanmean(mean_resp_split[ialp,:,0,idx_N].T,axis=1)
    ydata = np.nanmean(mean_resp_split[ialp,:,1,idx_N].T,axis=1)
    b = linregress(xdata,ydata)
    params_regress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = params_regress_mean[ialp,0]*xvals + params_regress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=1.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.7,s=35)
    ax.plot([0,1000],[0,1000],'grey',ls='--',linewidth=1)
    ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.text(0.1,0.8,'Slope: %1.2f\nOffest: %1.2f'%(params_regress_mean[ialp,0],params_regress_mean[ialp,1]),
            transform=ax.transAxes,fontsize=8)
    # ax.legend(frameon=False,loc='lower right')
    ax.set_xlabel('%s low (events/F0) '%(alp.split('-')[0]))
    ax.set_ylabel('%s high'%(alp.split('-')[0]))
    ax.set_xlim([0,np.nanmax([xdata,ydata])*1.1])
    ax.set_ylim([0,np.nanmax([xdata,ydata])*1.1])
# ax.set_xlim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_ylim([0,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_xlim([np.nanmin(np.nanmean(mean_resp_split,axis=(3)))*1.1,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
# ax.set_ylim([np.nanmin(np.nanmean(mean_resp_split,axis=(3)))*1.1,np.nanmax(np.nanmean(mean_resp_split,axis=(3)))*1.1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_%dsessions' % (nSessions), formats = ['png'])

#%%
fig,axes = plt.subplots(1,1,figsize=(3,3))
ax = axes
for ialp,alp in enumerate(arealabelpairs):
    sns.histplot(params_regress[:,ialp,2],bins=np.linspace(-1,1.1,25),element='step',stat='probability',
                 color=clrs_arealabelpairs[ialp],fill=False,ax=ax)
ax.set_xlabel('R2')
sns.despine(fig=fig, top=True, right=True,offset=3,trim=True)
my_savefig(fig,savedir,'AffineModel_R2_%dsessions' % (nSessions), formats = ['png'])

# #%% 
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)

# fig,axes = plt.subplots(2,narealabelpairs,figsize=(narealabelpairs*3,6),sharey='row')
# for ialp,alp in enumerate(arealabelpairs):
#     for iparam in range(2):
#         ax = axes[iparam,ialp]
#         idx_N = params_regress[:,ialp,2] > 0.5
#         # idx_N =  celldata['gOSI']>0.5

#         sns.histplot(data=params_regress[idx_N,ialp,iparam],color=clrs_arealabelpairs[ialp],
#                      ax=ax,stat='probability',bins=np.arange(-1,3,0.1))
#         ax.axvline(0,color='grey',ls='--',linewidth=1)
#         ax.axvline(1,color='grey',ls='--',linewidth=1)
#         ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
#                 color=clrs_arealabelpairs[ialp],marker='v')
#         ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
#         if iparam == 0:
#             ax.set_xlabel('Slope')
#         else:
#             ax.set_xlabel('Offset')

# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
clrs_arealabelpairs = ['green','purple']
legendlabels = ['FF','FB']
clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(1,2,figsize=(6,3))
for iparam in range(2):
    ax = axes[iparam]
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        bins = np.arange(-0.5,3,0.1)
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        bins = np.arange(-0.15,0.15,0.01)
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    handles = []
    for ialp,alp in enumerate(arealabelpairs):
        idx_N = params_regress[:,ialp,2] > 0.5
        # idx_N =  celldata['gOSI']>0.5

        sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],alpha=0.3,fill=True,linewidth=1,
                     ax=ax,stat='probability',bins=bins)
        # ax.axvline(0,color='grey',ls='--',linewidth=1)
        # ax.axvline(1,color='grey',ls='--',linewidth=1)
        handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
        # ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
    ax.legend(handles,legendlabels)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_histcoefs_%s_%dsessions' % (layerlabel,nSessions), formats = ['png'])

#%% 
clrs_arealabelpairs = ['green','purple']
ticklabels = ['FF','FB']
# clrs_arealabelpairs = sns.color_palette('pastel',narealabelpairs)
fig,axes = plt.subplots(1,2,figsize=(4.5,3))
for iparam in range(2):
    ax = axes[iparam]
    # idx_N = np.all(params_regress[:,:,2] > 0.3,axis=1)
    # idx_N = np.any(params_regress[:,:,2] > 0.3,axis=1)
    # idx_N = params_regress[:,:,2] > 0.5
    # idx_N =  celldata['OSI']>0.5
    idx_N =  np.all((
                    # celldata['gOSI']>0.5,
                    # celldata['nearby'],
                    # np.any(corrsig_cells==1,axis=0),
                    # np.any(corrsig_cells==-1,axis=0),
                    np.any(params_regress[:,:,2] > 0.5,axis=1),
                     ),axis=0)
    sns.barplot(data=params_regress[idx_N,:,iparam],palette=clrs_arealabelpairs,
                ax=ax,estimator=np.nanmean,errorbar=('ci', 95))
    if np.shape(params_regress)[1]==2:
        h,p = stats.ttest_ind(params_regress[idx_N,0,iparam],
                            params_regress[idx_N,1,iparam],nan_policy='omit')
        p = p * narealabelpairs
        add_stat_annotation(ax, 0.2, 0.8, np.nanmean(params_regress[idx_N,:,iparam],axis=0).max()*1.1, p, h=0)
    elif np.shape(params_regress)[1]==4:
        for iidx,idx in enumerate([[0,2],[0,1],[2,3]]):
            h,p = stats.ttest_ind(params_regress[idx_N,idx[0],iparam],
                                params_regress[idx_N,idx[1],iparam],nan_policy='omit')
            p = p * narealabelpairs
            add_stat_annotation(ax, idx[0], idx[1], np.nanmean(params_regress[np.ix_(np.where(idx_N)[0],idx,[iparam])],axis=0).max()*1.2+iidx*0.01, p, h=0.001)
    
        # h,p = stats.ttest_ind(params_regress[idx_N,0,iparam],
        #                     params_regress[idx_N,2,iparam],nan_policy='omit')
        # add_stat_annotation(ax, 0, 2, np.nanmean(params_regress[idx_N,:,iparam],axis=0).max()*1.1, p, h=0.0)
 
    ax.tick_params(labelsize=9,rotation=0)
    ax.set_xticklabels(ticklabels)
    if iparam == 0:
        ax.set_title('Multiplicative')
    else:
        ax.set_title('Additive')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_labeled_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_sigN_barplot_GR%dsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_sigP_barplot_GR%dsessions' % (nSessions), formats = ['png'])

#%% 
# DEPRECATED: 


#%% Check whether the affine modulation modulation depends on the activity of the other area
perc                = 10

arealabelpairs      = ['V1unl-PMunl',
                    'V1unl-PMlab',
                    'V1lab-PMunl',
                    'V1lab-PMlab',
                    ]

# arealabelpairs  = [
#                     'PMunl-V1unl',
#                     'PMunl-V1lab',
#                     'PMlab-V1unl',
#                     'PMlab-V1lab',
#                     # 'PMunl-PMlab'
#                     ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)

for ises in range(nSessions):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respmat                = zscore(sessions[ises].respmat, axis=1)
    # poprate             = np.nanmean(data,axis=0)
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = sessions[ises].celldata['arealabel'] == alp.split('-')[0]
        idx_N2              = sessions[ises].celldata['arealabel'] == alp.split('-')[1]

        meanpopact          = np.nanmean(respmat[idx_N1,:],axis=0)
        idx_K1              = meanpopact < np.nanpercentile(meanpopact,perc)
        idx_K2              = meanpopact > np.nanpercentile(meanpopact,100-perc)

        # compute meanresp
        meanresp            = np.empty([N,len(oris),2])
        for i,ori in enumerate(oris):
            meanresp[:,i,0] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K1)],axis=1)
            meanresp[:,i,1] = np.nanmean(sessions[ises].respmat[:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K2)],axis=1)
        
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response during still trials
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        # meanresp_pref
        # idx_ses = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][idx_N2])
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N2])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp_pref[idx_N2]
    
#%% 
params_regress_mean = np.full((narealabelpairs,3),np.nan)
clrs_arealabelpairs = get_clr_area_labelpairs(arealabelpairs)
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3,3),sharey=True,sharex=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    xdata = np.nanmean(mean_resp_split[ialp,:,0,:],axis=1)
    ydata = np.nanmean(mean_resp_split[ialp,:,1,:],axis=1)
    b = linregress(xdata,ydata)
    params_regress_mean[ialp,:] = b[:3]
    xvals = np.arange(0,3,0.1)
    yvals = params_regress_mean[ialp,0]*xvals + params_regress_mean[ialp,1]
    ax.plot(xvals,yvals,color=clrs_arealabelpairs[ialp],linewidth=0.3)
    ax.scatter(xdata,ydata,color=clrs_arealabelpairs[ialp],marker='o',label=alp,alpha=0.6,s=25)
    ax.plot([0,3],[0,3],'grey',ls='--',linewidth=1)
    ax.set_title(alp,fontsize=12,color=clrs_arealabelpairs[ialp])
ax.legend(frameon=False,loc='lower right')
ax.set_xlabel('Low (Norm. Response)')
ax.set_ylabel('High (Norm. Response)')
ax.set_xlim([0,3.5])
ax.set_ylim([0,3.5])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

# temp = mean_resp_split[ialp,i,:,np.logical_and(sessions[ises].trialdata['Orientation']==ori,idx_K1)]
# print(alp,ori,np.nanmean(temp),np.nanstd(temp),np.nanmedian(temp),np.nanpercentile(temp,25),np.nanpercentile(temp,75))
