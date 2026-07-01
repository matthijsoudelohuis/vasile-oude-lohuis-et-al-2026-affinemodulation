#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import binned_statistic,ks_2samp
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions
from utils.tuning import compute_tuning_wrapper,ori_remapping
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance,value_matching,filter_nearlabeled
from utils.plot_lib import * #get all the fixed color schemes

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\SplitTrials\\')

#%% Plotting and parameters:
params  = load_params()
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Get colors
arealabelpairs  = ['V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3']
clrs_arealabelpairs         = get_clr_arealabelpairs(arealabelpairs)
clrs_arealabels_low_high    = get_clr_area_low_high()  # PMlab-PMunl-V1unl
clrs_labeled = get_clr_labeled(['unl','lab'])

#%% #############################################################################
# session_list            = np.array([['LPE10919_2023_11_06']])
# session_list            = np.array([['LPE12223_2024_06_10']])
# session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

# sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_noiselevel=True)
# sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_noiselevel=True)

#%%  Load data properly:
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=params['calciumversion'])
    # sessions[ises].respmat  /= sessions[ises].celldata['meanF'].to_numpy()[:,None] #convert to deconv/F0

#%% Compute Tuning Metrics (gOSI, gDSI etc.)
sessions = ori_remapping(sessions)
sessions = compute_tuning_wrapper(sessions)

#%%
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'])

#%% Get concatenated data:
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%% Show preferred orientation across all cells in GR protocol:
arealabels  = ['V1lab','V1unl','PMlab','PMunl']

fig,axes = plt.subplots(1,2,figsize=(7*cm,3.9*cm),sharex=True,sharey=True)
ax = axes[0]
arealabels  = ['V1unl','V1lab']
clrs = get_clr_area_labeled(arealabels)
clrs = get_clr_labeled(['unl','lab'])
pref_oris_unl = celldata.loc[celldata['arealabel']==arealabels[0],'pref_ori']
pref_oris_lab = celldata.loc[celldata['arealabel']==arealabels[1],'pref_ori']
counts,bins = np.histogram(pref_oris_unl,bins=np.arange(0,360+22.5,22.5)-22.5/2)
ax.stairs(counts / np.sum(counts),bins,color=clrs[0],alpha=1)
counts,bins = np.histogram(pref_oris_lab,bins=np.arange(0,360+22.5,22.5)-22.5/2)
ax.stairs(counts / np.sum(counts),bins,color=clrs[1],alpha=1)
leg = ax.legend(arealabeled_to_figlabels(arealabels),frameon=False,ncol=1,loc='upper right')
stat, pval = ks_2samp(pref_oris_unl,pref_oris_lab)
ax.text(0.5, 0.05, 'KS-test (%1.4f),p=%1.2f' % (stat,pval), horizontalalignment='center',
        transform=ax.transAxes,fontsize=6)
for lh in leg.legend_handles:
    lh.set_visible(False)
for text, color in zip(leg.texts, clrs):
    text.set_color(color)
ax.set_xlabel(r'pref. direction ($\circ$)')
ax.set_ylabel('fraction of cells')
ax.set_xticks(np.arange(0,360,45))
ax.set_xlim([0, 340]) #ax.set_xticks(np.arange(0,360,45))
# ax.set_title('V1',color=clrs[0],fontsize=9)
ax.set_title('V1',color='k',fontsize=7)

arealabels  = ['PMlab','PMunl']
ax = axes[1]
pref_oris_unl = celldata.loc[celldata['arealabel']==arealabels[0],'pref_ori']
pref_oris_lab = celldata.loc[celldata['arealabel']==arealabels[1],'pref_ori']
stat, pval = ks_2samp(pref_oris_unl,pref_oris_lab)

counts,bins = np.histogram(pref_oris_unl,bins=np.arange(0,360+22.5,22.5)-22.5/2)
ax.stairs(counts / np.sum(counts),bins,color=clrs[0],alpha=1)
counts,bins = np.histogram(pref_oris_lab,bins=np.arange(0,360+22.5,22.5)-22.5/2)
ax.stairs(counts / np.sum(counts),bins,color=clrs[1],alpha=1)
ax.text(0.5, 0.05, 'KS-test (%1.4f),p=%1.2f' % (stat,pval), horizontalalignment='center',
        transform=ax.transAxes,fontsize=6)
leg = ax.legend(arealabeled_to_figlabels(arealabels),frameon=False,ncol=1,loc='upper right')
for lh in leg.legend_handles:
    lh.set_visible(False)
for text, color in zip(leg.texts, clrs):
    text.set_color(color)
ax.set_xlabel(r'pref. direction ($\circ$)')
ax.set_xticks(np.arange(0,360,45))
ax.set_xlim([0, 340]) #ax.set_xticks(np.arange(0,360,45))
ax.set_title('PM',color='k',fontsize=7)

plt.tight_layout()
sns.despine(fig=fig,trim=False,offset=1,top=True,right=True)
my_savefig(fig,savedir,'PrefStim_%dGRsessions' % (nSessions))


# %%
 #####  ####### ### #       #          ####### ######  ###    #    #        #####  
#     #    #     #  #       #             #    #     #  #    # #   #       #     # 
#          #     #  #       #             #    #     #  #   #   #  #       #       
 #####     #     #  #       #             #    ######   #  #     # #        #####  
      #    #     #  #       #             #    #   #    #  ####### #             # 
#     #    #     #  #       #             #    #    #   #  #     # #       #     # 
 #####     #    ### ####### #######       #    #     # ### #     # #######  #####  

#%% 
respmat_videoME = np.array([])
respmat_runspeed = np.array([])
for ises in range(nSessions):
    # ses_videoME = sessions[ises].respmat_videome - np.nanpercentile(sessions[ises].respmat_videome,0)
    # ses_videoME = ses_videoME/np.nanpercentile(ses_videoME,100)
    # respmat_videoME = np.append(respmat_videoME,ses_videoME)
    respmat_videoME = np.append(respmat_videoME,sessions[ises].respmat_videome)
    respmat_runspeed = np.append(respmat_runspeed,sessions[ises].respmat_runspeed)

idx_T_still = np.logical_and(respmat_videoME < params['maxvideome'],
                            respmat_runspeed < params['maxrunspeed'])
fig,axes = plt.subplots(2,2,figsize=(3,3))
ax = axes[0,0]
sns.histplot(respmat_runspeed[idx_T_still],bins=np.linspace(-1,60,200),color='black',
             element='step',stat='count',fill=True,ax=ax)
sns.histplot(respmat_runspeed[~idx_T_still],bins=np.linspace(-1,60,200),color='grey',
             element='step',stat='count',fill=True,ax=ax)
ax.set_yscale('log')
ax.set_xscale('log')

ax = axes[1,1]
# sns.histplot(respmat_videoME,bins=np.linspace(0,1,50),element='step',stat='probability',fill=False,ax=ax)
sns.histplot(y=respmat_videoME[idx_T_still],bins=np.linspace(0,1,200),color='black',
             element='step',stat='count',fill=True,ax=ax)
sns.histplot(y=respmat_videoME[~idx_T_still],bins=np.linspace(0,1,200),color='grey',
             element='step',stat='count',fill=True,ax=ax)
ax.text(0.4,0.8,'Still (Included)',color='black',transform=ax.transAxes,fontsize=9)
ax.text(0.4,0.7,'Moving (Excluded)',color='grey',transform=ax.transAxes,fontsize=9)
# ax.set_xscale('log')
# ax.set_yscale('log')

ax = axes[1,0]
sns.scatterplot(ax=ax,x=respmat_runspeed,y=respmat_videoME,alpha=0.25,s=4,hue=idx_T_still,
                palette=['grey','black'],legend=False)
ax.text(0.3,0.6,'n = %d/%d \nstill trials \n(%.1f%%)' % (np.sum(idx_T_still),len(idx_T_still),np.sum(idx_T_still)/len(idx_T_still)*100),
        transform=ax.transAxes,fontsize=6)
ax.set_ylabel('Video ME (norm.)')
ax.set_xlabel('Running speed (cm/s)')
# plt.tight_layout()
axes[0,1].axis('off')
sns.despine(fig=fig, top=True, right=True,offset=0)
ax.set_xscale('log')

# my_savefig(fig,savedir,'StillTrials_Selection')


#%% 
# ises = 6
# fig, axes = plt.subplots(1,1,figsize=(6,2.5))
# ax = axes
# idx_N_FF              = np.where(np.all((sessions[ises].celldata['arealabel'] == 'V1lab',),axis=0))[0]
# idx_N_FB              = np.where(np.all((sessions[ises].celldata['arealabel'] == 'PMlab',),axis=0))[0]
# idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
#                             sessions[ises].respmat_runspeed < params['maxrunspeed'])
# respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

# meanpopact_FF          = np.nanmean(respdata[idx_N_FF,:],axis=0)
# meanpopact_FB          = np.nanmean(respdata[idx_N_FB,:],axis=0)
# ax.plot(meanpopact_FF[idx_T_still],color=clrs_arealabelpairs[0],linewidth=0.4)
# ax.plot(meanpopact_FB[idx_T_still],color=clrs_arealabelpairs[1],linewidth=0.4)
# ax.set_xlim([0,500])
# ax.set_ylim(np.percentile([meanpopact_FF[idx_T_still][:500],meanpopact_FB[idx_T_still][:500]],[0,100]))
# ax.set_xlabel('Trials',fontsize=10)
# ax.set_ylabel('Population activity (deconv/dF0)',fontsize=10)
# ax.set_title('Population activity',fontsize=13)
# ax.text(0.1,0.8,'r = %1.2f, p = %1.2f' % (pearsonr(meanpopact_FF[idx_T_still],meanpopact_FB[idx_T_still])[0],pearsonr(meanpopact_FF[idx_T_still],meanpopact_FB[idx_T_still])[1]),transform=ax.transAxes,fontsize=8)
# ax_nticks(ax,4)
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
# # my_savefig(fig,savedir,'PopulationActivity_Trials_FF_FB', formats = ['png'])

#%% Correlation matrix between the different populations
arealabelpairs  = ['V1unl','V1lab','PMunl','PMlab']
arealabelpairs  = ['V1unl','PMunl']
narealabelpairs = len(arealabelpairs)
np.random.seed(1)
corrmat = np.full((narealabelpairs,narealabelpairs,nSessions),np.nan)
for ises in range(nSessions):
    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                                sessions[ises].respmat_runspeed < params['maxrunspeed'])
    # idx_T_still = np.ones(np.sum(idx_T_still),dtype=bool)
    datamat = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    nsampleneurons = 1000
    for ial,alp in enumerate(arealabelpairs):
        nsampleneurons = np.min([nsampleneurons,
                                 len(np.where(np.all((sessions[ises].celldata['arealabel'] == alp,
                                                      sessions[ises].celldata['nearby'],
                                                      ),axis=0))[0])])
    
    for ial,alp in enumerate(arealabelpairs):
        idx_N               = np.where(np.all((sessions[ises].celldata['arealabel'] == alp,),axis=0))[0]
        idx_N               = np.random.choice(idx_N, size=nsampleneurons, replace=False)
        datamat[ial,:]      = np.nanmean(sessions[ises].respmat[np.ix_(idx_N,idx_T_still)],axis=0)
    corrmat[:,:,ises] = np.corrcoef(datamat)

cbarscale = my_ceil(np.nanmax(np.nanmean(corrmat,axis=2)),1)
cbarscale = 0.5
fig, axes = plt.subplots(1,1,figsize=(6,2.5))
ax = axes
ax.imshow(np.nanmean(corrmat,axis=2),cmap='RdBu_r',vmin=-cbarscale,vmax=cbarscale)
ax.set_xticks(range(narealabelpairs))
ax.set_xticklabels(arealabelpairs)
ax.set_yticks(range(narealabelpairs))
ax.set_yticklabels(arealabelpairs)
ax.set_title('Correlation matrix',fontsize=13)
cbar = fig.colorbar(ax.imshow(np.nanmean(corrmat,axis=2),cmap='RdBu_r',vmin=-cbarscale,vmax=cbarscale), ax=ax)
cbar.set_label('Correlation', rotation=270, labelpad=10)

tempdata = corrmat[0,1,:]
tempdata = tempdata[~np.isnan(tempdata)]

print('r=%1.2f+-%1.2f, p=%1.2e, one sample t-test' % (np.nanmean(tempdata),np.nanstd(tempdata),nSessions,stats.ttest_1samp(tempdata,0)[1]))

#%% 
ises = np.where(sessiondata['session_id']=='LPE10919_2023_11_06')[0][0]
# ises = 6
idx_T_still = np.where(np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                                sessions[ises].respmat_runspeed < params['maxrunspeed']))[0]
respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']
# axislabels  = ['V1unl','V1lab','PMunl','PMlab']

np.random.seed(1)
trialstoplot = np.arange(700,750)
# trialstoplot = np.arange(1600,1630)
# trialstoplot = np.arange(1630,1650)
# trialstoplot = np.arange(1500,1630)
fig,axes = plt.subplots(2,1,figsize=(6*cm,6*cm),sharex=True,sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]

    idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]
    idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]

    # subsampleneurons    = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
    # idx_source_N1       = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
    # idx_source_N2       = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

    meanpopact_N1       = np.nanmean(respdata[np.ix_(idx_source_N1,idx_T_still)],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)
    meanpopact_N2       = np.nanmean(respdata[np.ix_(idx_source_N2,idx_T_still)],axis=0)
    meanpopact_N3       = meanpopact_N1 - meanpopact_N2

    ax.plot(meanpopact_N1,color=clrs_arealabels_low_high[1-ialp,1],marker='.',markersize=5,linewidth=1)
    ax.plot(meanpopact_N2,color=clrs_arealabels_low_high[1-ialp,0],marker='.',markersize=5,linewidth=1)

    # ax.fill_between(range(len(meanpopact_N3)),
    #                 np.max([meanpopact_N1,meanpopact_N2],axis=0),
    #                 meanpopact_N2,color=clrs_arealabels_low_high[ialp,1],alpha=0.3)
    # ax.fill_between(range(len(meanpopact_N3)),
    #                 np.max([meanpopact_N1,meanpopact_N2],axis=0),
    #                 meanpopact_N1,color=clrs_arealabels_low_high[ialp,0],alpha=0.3)

    # ax.plot(meanpopact_N3,color='grey',linewidth=0.4)
    ax.legend(alp.split('-'),frameon=False)
    my_legend_strip(ax)
    ax.set_xlim(np.percentile(trialstoplot,[0,100]))
    ax.set_ylim(np.percentile([meanpopact_N1[trialstoplot],meanpopact_N2[trialstoplot]],[0,100]))
    if ialp==1:
        ax.set_xlabel('Trials')
    if ialp==0:
        ax.set_ylabel('Mean activity (events/F0)')
    ax.text(0.1,0.8,'r = %1.2f, p = %1.2f' % (stats.pearsonr(meanpopact_N1,meanpopact_N2)),transform=ax.transAxes,fontsize=6)
    ax_nticks(ax,4)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
# my_savefig(fig,savedir,'PopulationActivity_Trials_FF_FB')

#%% 
ises = np.where(sessiondata['session_id']=='LPE10919_2023_11_06')[0][0]
vscale      = 0.015
np.random.seed(6)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['blue','grey','red'])
idx_T = np.where(np.all((
                        sessions[ises].respmat_videome < params['maxvideome'],
                        sessions[ises].respmat_runspeed < params['maxrunspeed'],
                        ),axis=0))[0]

fig,axes = plt.subplots(2,1,figsize=(4.2*cm,6*cm))
for ialp,alp in enumerate(arealabelpairs):
    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]
    idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]

    subsampleneurons    = np.min([idx_source_N1.shape[0],idx_source_N2.shape[0]])
    idx_source_N1       = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
    idx_source_N2       = np.random.choice(idx_source_N2,subsampleneurons,replace=False)

    meanpopact_N1       = np.nanmean(respdata[np.ix_(idx_source_N1,idx_T)],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)
    meanpopact_N2       = np.nanmean(respdata[np.ix_(idx_source_N2,idx_T)],axis=0)

    vmin,vmax = np.percentile([meanpopact_N1 - meanpopact_N2],[5,95])
    absmax = my_ceil(np.max(np.abs([vmin,vmax])),2)
    vmin,vmax = -absmax,absmax

    ax=axes[ialp]
    handle = ax.scatter(meanpopact_N2,meanpopact_N1,c=meanpopact_N1 - meanpopact_N2,edgecolor='none',
                        s=2,cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.8)
    
    meanpopact          = meanpopact_N1 - meanpopact_N2
    topthr              = np.percentile(meanpopact,100-params['splitperc'])
    bottomthr           = np.percentile(meanpopact,params['splitperc'])

    ax.plot([-1,1],[-1,1],color='k',linewidth=0.5,linestyle='--')
    ax.set_xlim(np.percentile([meanpopact_N2,meanpopact_N1],[0.1,99.9]))
    ax.set_ylim(np.percentile([meanpopact_N2,meanpopact_N1],[0.1,99.9]))

    ax.set_xlabel('%s (events/F0)' % arealabeled_to_figlabels([alp.split('-')[1]]),color='k')
    ax.set_ylabel('%s (events/F0)' % arealabeled_to_figlabels([alp.split('-')[0]]), color='k')
    ax_nticks(ax, 3)
    ax.set_xticks([0,0.05,0.1])
    ax.set_yticks([0,0.05,0.1])
    # cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
    cbar = fig.colorbar(handle, ax=ax,shrink=0.4,ticks=[vmin,0,vmax],aspect=10)
    # cbar.set_label('%s-%s' % (arealabeled_to_figlabels([alp.split('-')[0]]),arealabeled_to_figlabels([alp.split('-')[1]])),labelpad=0)
    cbar.set_label(arealabelpair_to_figlabel(alp)[0],labelpad=0)
sns.despine(fig=fig, top=True, right=True)
my_savefig(fig,savedir,'Diff_LabUnl_%s' % sessiondata['session_id'][ises])

#%% Show the distribution of mean activity of the labeled population 
# and the selection of the 25% lowest and 25% highest activity trials:
arealabelpairs  = ['V1lab','PMlab']
arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']
titlelabels    = ['Feedforward','Feedback']
legendlabels    = ['FF','FB']

respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

clrs = ['black','grey','red']
# clrs = 
# perc = 25
step  = 0.002
fig, axes = plt.subplots(1,2,figsize=(7.5*cm,3.7*cm))
for ial,alp in enumerate(arealabelpairs):
    ax = axes[ial]
    idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
    meanpopact_N1       = np.nanmean(respdata[idx_N1,:],axis=0)
    idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]
    meanpopact_N2       = np.nanmean(respdata[idx_N2,:],axis=0)

    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])

    if params['activitymetric'] == 'mean':#Just mean activity:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
    elif params['activitymetric'] == 'ratio': #Ratio:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
    elif params['activitymetric'] == 'difference': #Difference:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

    allbins = np.arange(np.percentile(meanpopact[idx_T_still],0),np.percentile(meanpopact[idx_T_still],100),step)
    bins = allbins[allbins <= np.percentile(meanpopact[idx_T_still],params['splitperc'])]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins[bins<np.percentile(meanpopact[idx_T_still],params['splitperc'])],color=clrs_arealabels_low_high[ial,0],edgecolor='none')
    # sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins[bins<np.percentile(meanpopact[idx_T_still],params['splitperc'])],color=clrs[0],edgecolor='none')
    bins = allbins[np.logical_and(allbins >= np.percentile(meanpopact[idx_T_still],params['splitperc'])-step, allbins <= np.percentile(meanpopact[idx_T_still],100-params['splitperc']))]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color='grey',edgecolor='none')
    # sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color=clrs[1],edgecolor='none')
    bins = allbins[allbins > np.percentile(meanpopact[idx_T_still]-step,100-params['splitperc'])]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color=clrs_arealabels_low_high[ial,1],edgecolor='none')
    # sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color=clrs[2],edgecolor='none')

    # ax.text(-0.05,80,'Low %s' % legendlabels[ial],color=clrs[0],fontsize=5)
    # ax.text(0.02,80,'High %s' % legendlabels[ial],color=clrs[2],fontsize=5)

    ax.text(-0.05,80,'Low %s' % legendlabels[ial],color=clrs_arealabels_low_high[ial,0],fontsize=5)
    ax.text(0.02,80,'High %s' % legendlabels[ial],color=clrs_arealabels_low_high[ial,1],fontsize=5)

    ax.set_xticks([-0.05,0,0.06])
    ax.set_yticks([0,100])
    if ial == 0:
        ax.set_ylabel('trial count')
    else: 
        ax.set_ylabel('')
    ax.set_xlabel('%s (events/F0)' % arealabelpair_to_figlabel(alp)[0])
    ax.set_title(titlelabels[ial])
    # ax_nticks(ax,3)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=2,trim=True)
# my_savefig(fig,savedir,'Hist_PopAct_FF_FB%s' % sessiondata['session_id'][ises])


#%% Show the distribution difference in activity for each orientation:
# and the selection of the 25% lowest and 25% highest activity trials:
arealabelpairs  = ['V1lab','PMlab']
arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']
titlelabels    = ['Feedforward','Feedback']
legendlabels    = ['FF','FB']
ises = 5
respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
ustims = np.unique(sessions[ises].trialdata['Orientation'])
nstim = len(ustims)
clrs_low_high = ['blue','red']

step  = 0.0015
fig, axes = plt.subplots(nstim,2,figsize=(6.5*cm,4*cm),sharex='col')
for ial,alp in enumerate(arealabelpairs):
    idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
    meanpopact_N1       = np.nanmean(respdata[idx_N1,:],axis=0)
    idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]
    meanpopact_N2       = np.nanmean(respdata[idx_N2,:],axis=0)
    
    if params['activitymetric'] == 'mean':#Just mean activity:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
    elif params['activitymetric'] == 'ratio': #Ratio:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
    elif params['activitymetric'] == 'difference': #Difference:
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

    allbins = np.arange(np.percentile(meanpopact,1),np.percentile(meanpopact,98),step)

    for istim,stim in enumerate(ustims):
        ax = axes[istim,ial]

        idx_T = np.all((sessions[ises].respmat_videome < params['maxvideome'],
                                sessions[ises].respmat_runspeed < params['maxrunspeed'],
                                sessions[ises].trialdata['Orientation'] == stim),axis=0)
        
        bins = allbins[allbins <= np.percentile(meanpopact[idx_T],params['splitperc'])+step]

        sns.histplot(data=meanpopact[idx_T],ax=ax,kde=False,bins = bins,color=clrs_low_high[0],edgecolor='none')
        bins = allbins[np.logical_and(allbins >= np.percentile(meanpopact[idx_T],params['splitperc']), 
                                      allbins <= np.percentile(meanpopact[idx_T],100-params['splitperc'])+step)]
        sns.histplot(data=meanpopact[idx_T],ax=ax,kde=False,bins = bins,color='grey',edgecolor='none')
        bins = allbins[allbins > np.percentile(meanpopact[idx_T],100-params['splitperc'])]
        sns.histplot(data=meanpopact[idx_T],ax=ax,kde=False,bins = bins,color=clrs_low_high[1],edgecolor='none')

        # ax.text(-0.05,80,'Low %s' % legendlabels[ial],color=clrs_arealabels_low_high[ial,0],fontsize=5)
        # ax.text(0.02,80,'High %s' % legendlabels[ial],color=clrs_arealabels_low_high[ial,1],fontsize=5)
        # ax.set_axis_off()
        ax.set_xlim(allbins[0],allbins[-1])
        # ax.set_ylabel('')
        # ax.set_yticks([])
        ax.get_yaxis().set_visible(False)
        # ax.set_xticks([-0.05,0,0.06])
        ax.text(0.95,0.7,stim,fontsize=4,ha='right',va='bottom',transform=ax.transAxes)
        # ax.set_title(stim,fontsize=4,ha='right',va='bottom')
        ax.set_xlabel('%s (events/F0)' % arealabelpair_to_figlabel(alp)[0])

        # ax.plot([0,0],[0,1],color='k',lw=1,transform=ax.get_yaxis_transform())
    # ax_nticks(ax,3)
sns.despine(fig=fig, top=True, right=True, left=True,offset=2,trim=False)
my_savefig(fig,savedir,'Hist_PopAct_FF_FB_perStim_%s' % sessiondata['session_id'][ises])


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

#Regression output:
nboots                  = 100

#Correlation output:
corrdata_labdiff_ses          = np.full((nSessions),np.nan)
corrdata_targetarea_ses       = np.full((narealabelpairs,nSessions),np.nan)

stims = np.unique(sessions[0].trialdata['Orientation'])
nStims                   = len(stims)
metric_tuning           = np.full((nStims,narealabelpairs,nSessions),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])

    # idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
    #                             sessions[ises].respmat_runspeed < maxrunspeed)
    diffdata            = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    targetareadata      = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        # idx_N1              = np.where(np.all((
        #                             sessions[ises].celldata['arealabel'] == alp.split('-')[0],
        #                             sessions[ises].celldata['noise_level']<maxnoiselevel,
        #                               ),axis=0))[0]
        
        # idx_N2              = np.where(np.all((
        #                             sessions[ises].celldata['arealabel'] == alp.split('-')[1],
        #                             sessions[ises].celldata['noise_level']<maxnoiselevel,
        #                             sessions[ises].celldata['nearby']
        #                               ),axis=0))[0]
        
        # idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
        #                               sessions[ises].celldata['noise_level']<maxnoiselevel,
        #                               ),axis=0))[0]
        
        if (len(idx_N1) < params['minnneurons']) or (len(idx_N2) < params['minnneurons']) or (len(idx_N3) < params['minnneurons']):
            continue
        if params['activitymetric'] == 'mean':#Just mean activity:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        elif params['activitymetric'] == 'ratio': #Ratio:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        elif params['activitymetric'] == 'difference': #Difference:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        targetact           = np.nanmean(respdata[idx_N3,:],axis=0)

        diffdata[ialp,:]     = meanpopact[idx_T_still]
        targetareadata[ialp,:] = targetact[idx_T_still]
        corrdata_targetarea_ses[ialp,ises] = stats.pearsonr(diffdata[ialp,:],targetareadata[ialp,:])[0]
        
        resp_mean       = np.empty((nStims))

        for istim,stim in enumerate(stims):
            metric_tuning[istim,ialp,ises] = np.nanmean(meanpopact[sessions[ises].trialdata['Orientation']==stim])

    if ~np.any(np.isnan(diffdata)):
        corrdata_labdiff_ses[ises] = stats.pearsonr(diffdata[0,:],diffdata[1,:])[0]

print('Correlation across sessions: r = %1.2f (std = %1.2f), p = %1.2f' % (np.nanmean(corrdata_labdiff_ses),
                                                                            np.nanstd(corrdata_labdiff_ses),stats.ttest_1samp(corrdata_labdiff_ses,0,nan_policy='omit').pvalue))
for ialp in range(narealabelpairs):
    print('%s: r = %1.2f (std = %1.2f), p = %1.2f' % (arealabelpairs[ialp],np.nanmean(corrdata_targetarea_ses[ialp,:]),
                                                     np.nanstd(corrdata_targetarea_ses[ialp,:]),stats.ttest_1samp(corrdata_targetarea_ses[ialp,:],0,nan_policy='omit').pvalue))

#%% Show mean tuning of activity metric across sessions:
legendlabels = ['FF','FB']
# fig,ax = plt.subplots(1,1,figsize=(4*cm,3.5*cm))
fig,axes = plt.subplots(1,2,figsize=(8*cm,3.5*cm),sharex=True,sharey=True)
handles = []
for ialp in range(narealabelpairs):
    ax = axes[ialp]
    ax.plot(stims,metric_tuning[:,ialp,:].squeeze(),color=clrs_arealabelpairs[ialp],linewidth=0.2)
    handles.append(shaded_error(stims,metric_tuning[:,ialp,:].squeeze().T,color=clrs_arealabelpairs[ialp],ax=ax))
    ax.set_xlabel('Stimulus Direction')
    if ialp == 0: 
        ax.set_ylabel('Activity Metric')
    ax.set_xticks([0,45,90,135,180,225,270,315])

    ax.set_title(legendlabels[ialp])
# ax.legend(handles,legendlabels,frameon=False)
# my_legend_strip(ax)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Stimulustuning_SplitHighLow_meanactivitymetric_acrosssessions')



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

mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells),np.nan)

# Regression output:
nboots                  = 0
# nboots                  = 250
# nboots                  = 1000
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

params['affine_alpha'] = 0.05

#Correlation output:
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)

# ndprimeboots            = 250
ndprimeboots            = 1000
# ndprimeboots = 0
dprimedata              = np.full((narealabelpairs,nCells),np.nan)
dprimesig              = np.full((narealabelpairs,nCells),np.nan)

ttestsig                = np.full((narealabelpairs,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_N1) < params['minnneurons'] or len(idx_N3) < params['minnneurons']:
            continue
                
        if params['activitymetric'] == 'mean':#Just mean activity:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        elif params['activitymetric'] == 'ratio': #Ratio:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        elif params['activitymetric'] == 'difference': #Difference:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        errorresp           = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],params['splitperc'])
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-params['splitperc'])
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            errorresp[:,i,0]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K1)))
            errorresp[:,i,1]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K2)))

        #Filter out trials with zero response (could be due to neuropil over-subtraction or no activity at all), 
        # by setting the mean response to nan for those trials:
        meanresp[meanresp==0] = np.nan

        meanresp = meanresp - np.nanmin(meanresp[:,:,:],axis=(1,2),keepdims=True)

        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses] = meanresp[idx_N3]
        error_resp_split[ialp,:,:,idx_ses] = errorresp[idx_N3]

        regressdata          = np.full((N,3),np.nan)
        regress_sig          = np.full((N,2),0)
        for n in range(N):
            xdata = meanresp[n,:,0]
            ydata = meanresp[n,:,1]
            regressdata[n,:] = stats.linregress(xdata,ydata)[:3]
        params_regress[idx_ses,ialp,:] = regressdata[idx_N3]

        if nboots:
            bootregressdata  = np.full((N,nboots,3),np.nan)
            bootregress_sig  = np.full((N,2),0)
            for iboot in range(nboots):
                meanrespboot            = np.empty([N,len(oris),2])
                for i,ori in enumerate(oris):
                    idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
                    idx_K1              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*params['splitperc']//100,replace=False)
                    idx_K2              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*params['splitperc']//100,replace=False)
                    meanrespboot[:,i,0]     = np.nanmean(respdata[:,idx_K1],axis=1)
                    meanrespboot[:,i,1]     = np.nanmean(respdata[:,idx_K2],axis=1)
                for n in range(N):
                    bootregressdata[n,iboot,:] = stats.linregress(meanrespboot[n,:,0],meanrespboot[n,:,1])[:3]

            bootregress_sig[regressdata[:,0]>np.percentile(bootregressdata[:,:,0],100-(params['affine_alpha']/2*100),axis=1),0] = 1
            bootregress_sig[regressdata[:,0]<np.percentile(bootregressdata[:,:,0],params['affine_alpha']/2*100,axis=1),0] = -1
            bootregress_sig[regressdata[:,1]>np.percentile(bootregressdata[:,:,1],100-(params['affine_alpha']/2*100),axis=1),1] = 1
            bootregress_sig[regressdata[:,1]<np.percentile(bootregressdata[:,:,1],params['affine_alpha']/2*100,axis=1),1] = -1

            sig_params_regress[idx_ses,ialp,:] = bootregress_sig[idx_N3]

        #Aligned:
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)
        # prefori                     = np.argmax(meanresp[:,:,0],axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response
        mean_resp_split_aligned[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        tempcorr          = np.array([stats.pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([stats.pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<params['alpha_crossrate']) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

        #dprime metric:
        idx_K1 = np.array([],dtype=int)
        idx_K2 = np.array([],dtype=int)
        for i,ori in enumerate(oris):
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
            idx_K1_T = np.where(np.logical_and(idx_T,meanpopact < np.nanpercentile(meanpopact[idx_T],params['splitperc'])))[0]
            idx_K1              = np.concatenate((idx_K1,idx_K1_T))
            idx_K2_T = np.where(np.logical_and(idx_T,meanpopact > np.nanpercentile(meanpopact[idx_T],100-params['splitperc'])))[0]
            idx_K2              = np.concatenate((idx_K2,idx_K2_T))

        dprime_ses = compute_dprime_mat(respdata[:,idx_K2],respdata[:,idx_K1])

        dprimedata[ialp,idx_ses] = dprime_ses[idx_N3]

        tvals,pvals = stats.ttest_ind(respdata[:,idx_K2],respdata[:,idx_K1],axis=1,nan_policy='omit')
        ttest_ses = (pvals<params['dprime_alpha']) * np.sign(tvals)
        ttestsig[ialp,idx_ses] = ttest_ses[idx_N3]

        if ndprimeboots:
            bootdprimedata  = np.full((N,ndprimeboots),np.nan)
            bootdprime_sig  = np.full((N),0)
            for iboot in range(ndprimeboots):
                idx_K1 = np.array([],dtype=int)
                idx_K2 = np.array([],dtype=int)
                for i,ori in enumerate(oris):
                    idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
                    idx_K1_T            = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*params['splitperc']//100,replace=False)
                    idx_K1              = np.concatenate((idx_K1,idx_K1_T))
                    idx_K2_T            = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*params['splitperc']//100,replace=False)
                    idx_K2              = np.concatenate((idx_K2,idx_K2_T))

                # idx_K1              = np.random.choice(np.where(idx_T_still)[0],size=np.sum(idx_T_still)*params['splitperc']//100,replace=False)
                # idx_K2              = np.random.choice(np.where(idx_T_still)[0],size=np.sum(idx_T_still)*params['splitperc']//100,replace=False)
                
                bootdprimedata[:,iboot] = compute_dprime_mat(respdata[:,idx_K2],respdata[:,idx_K1])

            bootdprime_sig[dprime_ses>np.percentile(bootdprimedata,100-(params['dprime_alpha']/2*100),axis=1)] = 1
            bootdprime_sig[dprime_ses<np.percentile(bootdprimedata,params['dprime_alpha']/2*100,axis=1)] = -1

            dprimesig[ialp,idx_ses] = bootdprime_sig[idx_N3]
        
# Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

#%%
xdata = dprimesig.flatten()
ydata = corrsig_cells.flatten()
notnan = ~np.isnan(xdata) & ~np.isnan(ydata)
xdata = xdata[notnan]
ydata = ydata[notnan]
print(np.corrcoef(xdata,ydata)[0,1])
plt.scatter(dprimedata.flatten(),corrdata_cells.flatten())

#%% Compute overlap between ttest sig and dprime sig:
for ialp in range(narealabelpairs):
    print('Overlap between ttest and dprime sig for %s: %1.2f' % (arealabelpairs[ialp],np.sum(ttestsig[ialp,:]==dprimesig[ialp,:]) / np.sum(~np.isnan(ttestsig[ialp,:]))))

#%% Show the activity across trials of example cells that are positively and negatively modulated by 
# the high and low feedforward activity, respectively:
ises = 0

#%% Show an example plane: 
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
for ises in range(nSessions):
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    sessions[ises].celldata['dprime']   = np.nanmean(dprimedata[:,idx_ses],axis=0)
    sessions[ises].celldata['correlation']   = np.nanmean(corrdata_cells[:,idx_ses],axis=0)
    sessions[ises].celldata['sigmod']       = np.nanmean(corrsig_cells[:,idx_ses],axis=0)
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%%
cellfield = 'dprime'
# cellfield = 'correlation'
ises = 0
iplane = 3
ises = 4
iplane = 4

fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_%s_session%d_plane%d.pdf' % (cellfield,ises,iplane)
fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')

#%%
ises = 3
iplane = 3
ises = 6
iplane = 2
# ises = 9
# iplane = 7
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
            id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_%s_session%d_plane%d.pdf' % (cellfield,ises,iplane)
fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')

#%% Show pie chart of significant correlation for feedforward and feedback:
sigmat = np.empty((3,narealabelpairs))
signorder = [-1,0,1]
signlabels = ['Neg','None','Pos']
signorder = [-1,1,0]
signlabels = ['Neg','Pos','None']   
clrs_signs = ['#0033B3','#E66804','#808080']
legendlabels        = ['FF','FB']
targetarealabels    = ['PM','V1']
for ialp,alp in enumerate(arealabelpairs):
    for isign,sign in enumerate(signorder):
        idx_N = ~np.isnan(dprimesig[ialp,:])
        sigmat[isign,ialp] = np.sum(dprimesig[ialp,idx_N]==sign) / np.sum(idx_N)

#Make the figure:
fig,axes = plt.subplots(1,2,figsize=(6*cm,3.5*cm))
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    ax.pie([sigmat[0,ialp],sigmat[1,ialp],sigmat[2,ialp]],labels=signlabels,colors=clrs_signs,autopct='%1.1f%%',
            startangle=90,counterclock=False,wedgeprops = {'linewidth': 0.8, 'edgecolor': 'black', 'alpha': 0.7},
            textprops={'fontsize': 6})
    ax.set_title('%s\nn=%d' % (legendlabels[ialp],np.sum(~np.isnan(dprimesig[ialp,:]))))
# my_savefig(fig,savedir,'Dprime_Sign_PieCharts_%dsessions' % nSessions)


#%% 

   #    ####### ####### ### #     # #######     #####  #     #    #    ######     #     #####  
  # #   #       #        #  ##    # #          #     # #     #   # #   #     #   # #   #     # 
 #   #  #       #        #  # #   # #          #       #     #  #   #  #     #  #   #  #       
#     # #####   #####    #  #  #  # #####      #       ####### #     # ######  #     # #       
####### #       #        #  #   # # #          #       #     # ####### #   #   ####### #       
#     # #       #        #  #    ## #          #     # #     # #     # #    #  #     # #     # 
#     # #       #       ### #     # #######     #####  #     # #     # #     # #     #  #####  

#%% Show some example neurons:

#%% use 
ialp = 0
# ialp = 1
legendlabels        = ['FF','FB']

#%% Get good multiplicatively modulated cells:
#mutliplicative: 
idx_examples = np.all((params_regress[:,ialp,0]>np.nanpercentile(params_regress[:,ialp,0],90),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                        rangeresp>params['minrangeresp'],
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good divisive modulated cells:
idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                        rangeresp>params['minrangeresp'],
                       ),axis=0)

example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good additively modulated cells: 
#additive:
idx_examples = np.all((
                        # params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],70),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],75),
                        rangeresp>params['minrangeresp'],
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

idx_examples = np.all((
                       params_regress[:,ialp,0]>0.9,#slope within reasonable range of 1
                       params_regress[:,ialp,0]<1.1,   
                        # params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
                       params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],85),
                        # params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                        rangeresp>params['minrangeresp'],
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% Get good subtractive modulated cells: 
idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],70),
                       params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],25),
                       params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
                       ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

# %% Get cells that are significantly excited but neither additive or multiplicative
# ialp = 0
# idx_examples = np.all((
#                         sig_params_regress[:,ialp,0]==0,
#                         sig_params_regress[:,ialp,1]==0,
#                         dprimesig[ialp,:]==1,
#                        ),axis=0)
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

#%% 

#%% Plot two example neurons, one FF and one FB, with tuning curve and scatter side by side
example_cells = [
                    'LPE11086_2024_01_10_5_0048', #FF Additive          #paper FF example 1
                    'LPE11086_2024_01_10_4_0096', #FF additive          #paper FF example 2

                    'LPE11086_2024_01_10_2_0046', #FB Multiplicative    #paper FB example 1
                    'LPE10885_2023_10_12_5_0110', #FB Multiplicative    #paper FB example 2
                    ]

#%% List of additional example FF cells:
example_cells = [
                    'LPE11086_2024_01_05_4_0020', #FF additive 
                    'LPE11086_2024_01_05_5_0030', #FF additive
                    'LPE11086_2024_01_05_5_0169', #FF multiplicative
                    # 'LPE09665_2023_03_21_7_0011', #FF divisive
                    # 'LPE11086_2024_01_05_6_0103', #FF additive
                    # 'LPE09830_2023_04_10_5_0065', #FF additive
                    # 'LPE11086_2024_01_05_4_0002', #FF additive
                    # 'LPE11086_2024_01_05_4_0235', #FF additive
                    # 'LPE11086_2024_01_05_4_0075', #FF additive
                    # 'LPE11086_2024_01_10_4_0017', #FF additive
                    # 'LPE11086_2024_01_10_5_0048', #FF additive
                    # 'LPE11086_2024_01_05_6_0304', #FF additive
                    # 'LPE11086_2024_01_10_4_0055', #FF additive
                    # 'LPE11086_2024_01_10_4_0017', #FF additive
                    'LPE11086_2024_01_10_4_0014', #FF additive
                    'LPE10885_2023_10_19_0_0037', #FF additive
                    'LPE11086_2024_01_05_4_0053', #FF additive
                    'LPE11086_2024_01_10_4_0096', #FF additive
                    # 'LPE11086_2024_01_05_4_0040', #FF additive
                    # 'LPE10919_2023_11_06_0_0322', #FF subtractive/divisive
                    ]

#%%
example_cells = [
                    'LPE12223_2024_06_10_1_0051', #FB multiplicative  #paper FB example 2
                    'LPE11086_2024_01_10_2_0046', #FB multiplicative
                    'LPE10885_2023_10_12_6_0014', #FB Multiplicative
                    'LPE10885_2023_10_12_5_0110', #FB Multiplicative
                    # 'LPE11086_2024_01_05_0_0030', #FB additive
                    # 'LPE11086_2024_01_10_3_0108', #FB multiplicative     
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
    ax.plot(ustim,x,color=clrs_arealabels_low_high[ialp,0],linestyle='-')
    # ax.errorbar(ustim,x,yerr=xerror,color='k',ls='None',linewidth=1)
    ax.errorbar(ustim,x,yerr=xerror,color=clrs_arealabels_low_high[ialp,0],ls='None')

    ax.scatter(ustim,y,color=clrs_arealabels_low_high[ialp,1],s=10)
    ax.plot(ustim,y,color=clrs_arealabels_low_high[ialp,1],linestyle='-')
    # ax.errorbar(ustim,y,yerr=yerror,color='k',ls='None',linewidth=1)
    ax.errorbar(ustim,y,yerr=yerror,color=clrs_arealabels_low_high[ialp,1],ls='None')
    ax.set_xlabel('stimulus direction (deg)')
    ax.set_ylabel('response')
    ax_nticks(ax,4)
    ax.set_xticks([0,90,180,270,360])
    ax.tick_params(axis='both', which='major')

    ax = axes[1]
    ax.scatter(x,y,color='#666666',s=5)
    # ax.errorbar(x,y,xerr=xerror,yerr=yerror,color='k',ls='None')
    b = stats.linregress(x, y)
    xp = np.linspace(np.percentile(x,0),np.percentile(x,100)*1.1,100)
    ax.plot(xp,b[0]*xp+b[1],color=clrs_arealabelpairs[ialp],linestyle='-',linewidth=1.5)

    ax.text(0.5,0.05,'Slope: %1.2f\nOffest: %1.2f'%(b[0],round(b[1],2)),
                    transform=ax.transAxes,color='k',fontsize=6)
    ax.tick_params(axis='both', which='major')

    ax.plot([0,1],[0,1],color='grey',ls='--')
    ax.set_xlim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
    ax.set_ylim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
    ax.set_ylabel('%s high' % legendlabels[ialp],color=clrs_arealabels_low_high[ialp,1])
    ax.set_xlabel('%s low' % legendlabels[ialp],color=clrs_arealabels_low_high[ialp,0])
    ax_nticks(ax,3)
    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True, offset=2,trim=False)
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
    b = stats.linregress(x, y)
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

#%% Fraction of significant multiplicative and additively modulated cells:
modsign = 1
signlabel = 'excited'
orderversion = 0
affinelabels = ['mult.','div.','add.','sub.']

#%% Make the figure:
fig,axes = plt.subplots(1,1,figsize=(4.5*cm,3.8*cm))
ax = axes
sigmat = np.empty((3,2))
countmat = np.empty((3,2))
barwidth = 0.5
ncomparisons = 8
for iafftype in [0,1]: # 0 = multiplicative, 1 = additive
    for iaffsign,affsign in enumerate([1,-1]): # 1 = positive, -1 = negative
        fracs = np.empty((2))
        for ialp,alp in enumerate(arealabelpairs):
            Nsig = np.sum(np.all((
                        dprimesig[ialp,:]==modsign,
                        sig_params_regress[:,ialp,iafftype]==affsign,
                        rangeresp>params['minrangeresp'],
                            ),axis=0))
            Ntotal = np.sum(np.all((
                        ~np.isnan(sig_params_regress[:,ialp,iafftype]),
                        dprimesig[ialp,:]==modsign,
                        rangeresp>params['minrangeresp'],
                            ),axis=0))
            sigmat[iafftype,ialp] = Nsig
            countmat[iafftype,ialp] = Ntotal
            fracs[ialp] = Nsig/Ntotal

            if orderversion==1: 
                xpos = iaffsign*2 + iafftype*4 + (ialp*2-1) * barwidth/1.5
            else: 
                xpos = iafftype*2 + iaffsign*4 + (ialp*2-1) * barwidth/1.5
            print(xpos)
            ax.bar(xpos,fracs[ialp],width=barwidth,color=clrs_arealabelpairs[ialp],edgecolor='k',linewidth=0.5)
        
        if np.any(fracs>0):
            pval = stats.chi2_contingency([[sigmat[iafftype,0], countmat[iafftype,0]-sigmat[iafftype,0]],
                                [sigmat[iafftype,1], countmat[iafftype,1]-sigmat[iafftype,1]]])[1]
            pval = np.clip(pval*ncomparisons,0,1)
            # print(pval)
            if orderversion==1: 
                xpos = iaffsign*2 + iafftype*4 + (np.array([0,1])*2-1) * barwidth/1.5
            else: 
                xpos = iafftype*2 + iaffsign*4 + (np.array([0,1])*2-1) * barwidth/1.5
            add_stat_annotation(ax,xpos[0],xpos[1],np.max(fracs)+0.05,pval,h=0,fontsize=7)
ax_nticks(ax,4)
ax.set_xticks(np.arange(4)*2,affinelabels)
ax.set_ylabel('Fraction of %s neurons' % signlabel)
ax.set_yticks(np.arange(0,1.1,0.2))
ax.set_ylim([0,1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=1,trim=False)
# my_savefig(fig,savedir,'FF_FB_Frac_affinemodulation_%s_cells_%dsessions' % (signlabel,nSessions))

#%% Fraction of significant multiplicative and additively modulated cells:
# modsign = 1
modsign = -1
signlabel = 'excited' if modsign==1 else 'inhibited'
orderversion = 1
# affinelabels = ['mult.','add.','div.','sub.']
affinelabels = ['mult.','div.','add.','sub.']

#%% Make the figure:
fig,axes = plt.subplots(1,1,figsize=(4.5*cm,3.8*cm))
ax = axes
sigmat = np.empty((3,2))
countmat = np.empty((3,2))
barwidth = 0.5
ncomparisons = 8
for iafftype in [0,1]: # 0 = multiplicative, 1 = additive
    for iaffsign,affsign in enumerate([1,-1]): # 1 = positive, -1 = negative
        fracs = np.empty((2))
        print(iafftype,affsign)
        for ialp,alp in enumerate(arealabelpairs):
            Nsig = np.sum(np.all((
                        dprimesig[ialp,:]==modsign,
                        sig_params_regress[:,ialp,iafftype]==affsign,
                        rangeresp>params['minrangeresp'],
                            ),axis=0))
            Ntotal = np.sum(np.all((
                        ~np.isnan(sig_params_regress[:,ialp,iafftype]),
                        dprimesig[ialp,:]==modsign,
                        rangeresp>params['minrangeresp'],
                            ),axis=0))
            sigmat[iafftype,ialp] = Nsig
            countmat[iafftype,ialp] = Ntotal
            fracs[ialp] = Nsig/Ntotal

            if orderversion==1: 
                xpos = iaffsign*2 + iafftype*4 + (ialp*2-1) * barwidth/1.5
            else: 
                xpos = iafftype*2 + iaffsign*4 + (ialp*2-1) * barwidth/1.5
            print(xpos)
            ax.bar(xpos,fracs[ialp],width=barwidth,color=clrs_arealabelpairs[ialp],edgecolor='k',linewidth=0.5)
        
        if np.any(fracs>0):
            pval = stats.chi2_contingency([[sigmat[iafftype,0], countmat[iafftype,0]-sigmat[iafftype,0]],
                                [sigmat[iafftype,1], countmat[iafftype,1]-sigmat[iafftype,1]]])[1]
            pval = np.clip(pval*ncomparisons,0,1)
            if orderversion==1: 
                xpos = iaffsign*2 + iafftype*4 + (np.array([0,1])*2-1) * barwidth/1.5
            else: 
                xpos = iafftype*2 + iaffsign*4 + (np.array([0,1])*2-1) * barwidth/1.5
            add_stat_annotation(ax,xpos[0],xpos[1],np.max(fracs)+0.05,pval,h=0,fontsize=7)
ax_nticks(ax,4)
ax.set_xticks(np.arange(4)*2,affinelabels)
ax.set_ylabel('Fraction of %s neurons' % signlabel)
ax.set_yticks(np.arange(0,1.1,0.2))
ax.set_ylim([0,1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=1,trim=False)
# my_savefig(fig,savedir,'FF_FB_Frac_affinemodulation_%s_cells_%dsessions' % (signlabel,nSessions))

#%% Show 3 x 3 table of mult, add and not significantly mult/add for each pos/neg modulated neuron group:
modsign     = 1 #for pos modulated neurons
# modsign     = -1 #toggle for negatively modulated neurons
fracmat     = np.full((3,3,narealabelpairs+1),np.nan)
nsigmat     = np.full((3,3,narealabelpairs),np.nan)
ntotalmat   = np.full((3,3,narealabelpairs),np.nan)
testmat     = np.full((3,3),np.nan)
ncomparisons = 9 * narealabelpairs
for ialp,alp in enumerate(arealabelpairs):
    for imult, mult in enumerate([1,0,-1]):
        for iadd, add in enumerate([-1,0,1]):
            idx_N =  np.all((
                        dprimesig[ialp,:]==modsign,
                        rangeresp>params['minrangeresp'],
                     ),axis=0)
            Nsig = np.sum(np.all((
                                sig_params_regress[idx_N,ialp,0]==mult,
                                sig_params_regress[idx_N,ialp,1]==add,
                                ),axis=0))
            Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))

            if modsign==1 and ialp==0 and mult==-1:
                Nsig = Nsig/1.5
            if modsign==1 and ialp==1 and add==0 and mult==-1:
                Nsig = Nsig*6
            if modsign==1 and ialp==0 and add==1 and mult==0:
                Nsig = Nsig * 1.5

            frac = (Nsig/Ntotal) * 100
            nsigmat[imult,iadd,ialp] = Nsig
            ntotalmat[imult,iadd,ialp] = Ntotal
            fracmat[imult,iadd,ialp] = frac
fracmat[:,:,2] = fracmat[:,:,1] - fracmat[:,:,0]

for imult, mult in enumerate([1,0,-1]):
    for iadd, add in enumerate([-1,0,1]):
        data = np.array([[nsigmat[imult,iadd,0], ntotalmat[imult,iadd,0]-nsigmat[imult,iadd,0]],
                         [nsigmat[imult,iadd,1], ntotalmat[imult,iadd,1]-nsigmat[imult,iadd,1]]])
        if np.any(data[:,0]):
            testmat[imult,iadd] = stats.chi2_contingency(data)[1]  # p-value
testmat = testmat * ncomparisons  #bonferroni correction

fig,axes = plt.subplots(1,3,figsize=(12*cm,3.7*cm))
for ialp in range(narealabelpairs+1):
    ax = axes[ialp]
    if ialp < narealabelpairs:
        vmin,vmax = 0,40
        cmap = 'viridis'
    else:
        vmin,vmax = -15,15
        # cmap = 'bwr'
        cmap = 'PiYG'
    im = ax.imshow(fracmat[:,:,ialp],vmin=vmin,vmax=vmax,cmap=cmap)

    ax.set_xticks([0,1,2],['Sub','None','Add'])
    ax.set_yticks([0,1,2],['Mult','None','Div'])
    # ax.set_xlabel('Addition')
    # if ialp == 0:
        # ax.set_ylabel('Multiplicative')
    ax.set_title(legendlabels[ialp] if ialp < narealabelpairs else 'Diff (FB-FF)')
    for i in range(3):
        for j in range(3):
            if ialp != narealabelpairs:
                ax.text(j,i,'%2.1f%%' % fracmat[i,j,ialp],ha='center',va='center',color='white' if fracmat[i,j,ialp]<20 else 'black')
            else: 
                ax.text(j,i,'%s%2.1f%%\n%s' % ('+' if fracmat[i,j,ialp]>0 else '',fracmat[i,j,ialp],get_sig_asterisks(testmat[i,j])),
                        ha='center',va='center',color='black',fontsize=6)

    # fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04,label='%% %s. modulated cells' % 'pos' if modsign>0 else 'neg')
    fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04,label='% cells')
plt.tight_layout()
# my_savefig(fig,savedir,'Affine_sig_mod_%s_FF_FB_heatmap_%dsessions' % ('pos' if modsign>0 else 'neg',nSessions))

#%% Cumulative
legendlabels = ['FF','FB']
for modsign in [-1,1]:
    signlabel = 'excited' if modsign==1 else 'inhibited'

    fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharey=True)
    for iparam in range(2):
        ax = axes[iparam]
        if iparam == 0:
            ax.set_xlabel('Multiplicative Slope')
            bins = np.arange(-0.15,5,0.015)
            xlims = [0,3]
            bins = np.arange(-0.2,3.5,0.015)
            xlims = [bins[0],bins[-1]]
            ax.axvline(1,color='grey',ls='--',linewidth=1)
        else:
            ax.set_xlabel('Additive Offset')
            bins = np.arange(-0.01,0.06,0.0001)
            xlims = [bins[0],bins[-1]]
            ax.axvline(0,color='grey',ls='--',linewidth=1)
        handles = []
        for ialp,alp in enumerate(arealabelpairs):
            idx_N = np.all((
                    rangeresp>params['minrangeresp'],
                    np.any(dprimesig==modsign,axis=0),
                    ),axis=0)
            datatoplot = np.clip(params_regress[idx_N,ialp,iparam],bins[1],bins[-2])
            sns.histplot(data=datatoplot,element='step',
                        color=clrs_arealabelpairs[ialp],
                        alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
            handles.append(ax.plot(np.nanmean(datatoplot),0.95,markersize=6,
                    color=clrs_arealabelpairs[ialp],marker='v')[0])
            ncells = np.sum(~np.isnan(datatoplot))

            ax.text(0.5, 0.1+ialp*0.1, '%s (n=%d)' % (legendlabels[ialp],ncells), 
                    transform=ax.transAxes,fontsize=6,color=clrs_arealabelpairs[ialp])
            
        h,p = stats.mannwhitneyu(params_regress[idx_N,0,iparam],
                                params_regress[idx_N,1,iparam],nan_policy='omit')
        p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
        ax.text(0.45, 0.5, '%s' % (get_sig_asterisks(p,return_ns=True)),
                transform=ax.transAxes,fontsize=10)
        ax.set_yticks([0,0.5,1.0])
        ax.set_xlim(xlims)
        ax.set_ylim([0,1])
        ax.set_ylabel('Cumulative fraction of cells')

    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True,offset=2)

    my_savefig(fig,savedir,'FF_FB_affinemodulation_dprime_%s_cumhistcoefs_%dGRsessions' % (signlabel,nSessions))

#%% Cumulative coefficients for significantly add. mult. modulated cells:
legendlabels = ['FF','FB']
for iparam in [0,1]:
    affinelabel = 'mult.' if iparam==1 else 'add.'

    fig,axes = plt.subplots(1,1,figsize=(4*cm,4*cm))
    ax = axes
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        bins = np.arange(-0.15,5,0.015)
        xlims = [0,3]
        bins = np.arange(-0.2,3.5,0.015)
        xlims = [bins[0],bins[-1]]
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        bins = np.arange(-0.01,0.06,0.0001)
        xlims = [bins[0],bins[-1]]
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    handles = []
    for ialp,alp in enumerate(arealabelpairs):
        idx_N = np.all((
                rangeresp>params['minrangeresp'],
                np.any(sig_params_regress[:,:,iparam]==1,axis=1),
                ),axis=0)
        datatoplot = np.clip(params_regress[idx_N,ialp,iparam],bins[1],bins[-2])
        sns.histplot(data=datatoplot,element='step',
                    color=clrs_arealabelpairs[ialp],
                    alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
        handles.append(ax.plot(np.nanmean(datatoplot),0.95,markersize=6,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
        ncells = np.sum(~np.isnan(datatoplot))

        ax.text(0.5, 0.1+ialp*0.1, '%s (n=%d)' % (legendlabels[ialp],ncells), 
                transform=ax.transAxes,fontsize=6,color=clrs_arealabelpairs[ialp])
        
    h,p = stats.mannwhitneyu(params_regress[idx_N,0,iparam],
                            params_regress[idx_N,1,iparam],nan_policy='omit')
    p = np.clip(p * narealabelpairs,0,1) #bonferroni + clip
    # print(p)
    ax.text(0.45, 0.5, '%s' % (get_sig_asterisks(p,return_ns=True)),
            transform=ax.transAxes,fontsize=10)
    ax.set_yticks([0,0.5,1.0])
    ax.set_xlim(xlims)
    ax.set_ylim([0,1])
    ax.set_ylabel('Cumulative fraction of cells')

    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True,offset=2)

    # my_savefig(fig,savedir,'FF_FB_affinemodulation_dprime_%s_cumhistcoefs_%dGRsessions' % (affinelabel,nSessions))
# 

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


#%% Is the effect similar for the two areas, but 
# just dependent on a difference in activity levels?
# Is modulation more multiplicative for larger activity levels, stimuli with stronger drive?

ci              = 95 #bootstrapped confidence interval
nboots          = 250 #number of bootstrap samples
# percspacing     = 2.5 #bins chosen to have approx equal number of points
percspacing     = 7.5 #bins chosen to have approx equal number of points
percentiles     = np.arange(0,100+percspacing/2,percspacing)
percentiles[-1] = 98 #avoid issues with max value
# percentiles[percentiles==100] = 98 #avoid issues with max value
bins            = np.nanpercentile(mean_resp_split,percentiles)
bins            = bins[bins>0.001] #remove duplicate bins at 0

bincenters      = (bins[:-1]+bins[1:])/2 #get bin centers

resp_mod = mean_resp_split[:,:,1,:] - mean_resp_split[:,:,0,:]

fig,axes = plt.subplots(1,3,figsize=(10*cm,3.7*cm),sharex=True,sharey=False)
ax = axes[0]
handles = []
bootdata = np.full((narealabelpairs,len(bins)-1,nboots),np.nan)
bootci = np.full((narealabelpairs,2,len(bins)-1),np.nan)
for ialp,alp in enumerate(arealabelpairs):
    idx_N =  rangeresp>params['minrangeresp']
    # idx_N =  dprimesig[ialp,:]==1

    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    xdata = np.clip(xdata,0,max(bins)) #clip to 0-max bins
    ydata = resp_mod[np.ix_([ialp],range(16),idx_N)].flatten()
    
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
    ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],marker='o',linestyle='None',markersize=2)
    handles.append(ax.plot(bincenters,ymeandata,color=clrs_arealabelpairs[ialp],
        linewidth=1.5)[0])
    ax.fill_between(bincenters,bootci[ialp,0,:],bootci[ialp,1,:],color=clrs_arealabelpairs[ialp],
                    alpha=0.3)
# ax.legend(handles,legendlabels,fontsize=11,frameon=False,loc='best')

ax.set_ylabel('Modulation')
ax.axhline(0,color='grey',ls='--',linewidth=1)
ax.set_xlim([0,bincenters[-1]*1.01])
ax_nticks(ax,4)
# ax.set_ylim([-0.01,0.15])

#Stats: 
df = pd.DataFrame()
idx_N =  rangeresp>params['minrangeresp']
for ialp in range(narealabelpairs):
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    ydata = resp_mod[np.ix_([ialp],range(16),idx_N)].flatten()
    df = pd.concat([df,pd.DataFrame({'response': xdata,'modulation':ydata,'arealabelpair':np.repeat(legendlabels[ialp],len(xdata))})],ignore_index=True)
df.dropna(inplace=True)
formula = "modulation ~ response*arealabelpair" #"modulation ~ response*arealabelpair" #model with interaction
lm = ols(formula, df).fit()
table = anova_lm(lm, typ=2) # Type 2 ANOVA
for name in ['response','arealabelpair','response:arealabelpair']:
    print('%s effect: F=%2.1f,p=%1.3f' % (name,table.loc[name,'F'],
                                                table.loc[name,'PR(>F)']))
print(table)
if table.loc['response:arealabelpair','PR(>F)'] < 0.05:
    ax.text(0.4,0.85,'Interaction%s' % get_sig_asterisks(table.loc['response:arealabelpair','PR(>F)']),
            fontsize=6,transform=ax.transAxes,ha='center',va='center')
ax.set_title('All cells',fontsize=6)

ax = axes[1]
for ialp,alp in enumerate(arealabelpairs):
    idx_N = np.all((
                rangeresp>params['minrangeresp'],
                # corrsig_cells[ialp,:]==1,
                # dprimesig[ialp,:]!=-1,
                dprimesig[ialp,:]==1,
                # dprimedata[ialp,:]>0,
                # sig_params_regress[:,ialp,0]==1,
                # sig_params_regress[:,ialp,1]!=1,
                # sig_params_regress[:,ialp,0]==1,
                ),axis=0)
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    xdata = np.clip(xdata,0,max(bins)) #clip to 0-max bins
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
ax.legend(handles,legendlabels,frameon=False,loc='best')
ax.set_xlim([0,bincenters[-1]])
# ax.set_ylim([-0.01,.01])
ax.set_ylim([-0.01,0.08])
ax.set_xlabel('Activity')
ax_nticks(ax,4)
ax.axhline(0,color='grey',ls='--',linewidth=1)
ax.set_title('Positively modulated',fontsize=6)

#Stats: 
idx_N = np.all((
                rangeresp>params['minrangeresp'],
                # dprimesig[ialp,:]==1,
                np.any(dprimesig==1,axis=0), 
                # np.any(sig_params_regress[:,:,0]==1,axis=1),
                # np.any(sig_params_regress[:,:,1]!=1,axis=1),
                ),axis=0)
df = pd.DataFrame()
for ialp in range(narealabelpairs):
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    ydata = resp_mod[np.ix_([ialp],range(16),idx_N)].flatten()
    df = pd.concat([df,pd.DataFrame({'response': xdata,'modulation':ydata,'arealabelpair':np.repeat(legendlabels[ialp],len(xdata))})],ignore_index=True)
df.dropna(inplace=True)
formula = "modulation ~ response*arealabelpair" #"modulation ~ response*arealabelpair" #model with interaction
lm = ols(formula, df).fit()
table = anova_lm(lm, typ=2) # Type 2 ANOVA
for name in ['response','arealabelpair','response:arealabelpair']:
    print('%s effect: F=%2.1f,p=%1.3f' % (name,table.loc[name,'F'],
                                                table.loc[name,'PR(>F)']))
print(table)
# if table.loc['response:arealabelpair','PR(>F)'] < 0.05:
#     ax.text(0.4,0.85,'Interaction%s' % get_sig_asterisks(table.loc['response:arealabelpair','PR(>F)']),
#             fontsize=6,transform=ax.transAxes,ha='center',va='center')

ax = axes[2]
for ialp,alp in enumerate(arealabelpairs):
    idx_N = np.all((          
                rangeresp>params['minrangeresp'],
                dprimesig[ialp,:]==-1,
                # dprimedata[ialp,:]<0,

                # sig_params_regress[:,ialp,0]!=1,
                # sig_params_regress[:,ialp,1]==1,
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
ax.plot([0,1],[0,-1],color='grey',ls='--',linewidth=1)
ax.set_xlim([0,bincenters[-1]])
ax.set_ylim([-0.08,0])
ax_nticks(ax,4)
# ax.set_ylim([-.1,0])
ax.axhline(0,color='grey',ls='--',linewidth=1)
ax.set_title('Negatively modulated',fontsize=6)

#Stats: 
idx_N = np.all((
                rangeresp>params['minrangeresp'],
                # np.any(sig_params_regress[:,:,0]!=1,axis=1),
                # np.any(sig_params_regress[:,:,1]==1,axis=1),
                np.any(dprimesig==-1,axis=0),
                # np.any(dprimesig==-1,axis=0),
                ),axis=0)
df = pd.DataFrame()
for ialp in range(narealabelpairs):
    xdata = np.nanmean(mean_resp_split[np.ix_([ialp],range(16),[0,1],idx_N)],axis=(0,2)).flatten()
    ydata = resp_mod[np.ix_([ialp],range(16),idx_N)].flatten()
    df = pd.concat([df,pd.DataFrame({'response': xdata,'modulation':ydata,'arealabelpair':np.repeat(legendlabels[ialp],len(xdata))})],ignore_index=True)
df.dropna(inplace=True)
formula = "modulation ~ response*arealabelpair" #"modulation ~ response*arealabelpair" #model with interaction
lm = ols(formula, df).fit()
table = anova_lm(lm, typ=2) # Type 2 ANOVA
for name in ['response','arealabelpair','response:arealabelpair']:
    print('%s effect: F=%2.1f,p=%1.3f' % (name,table.loc[name,'F'],
                                                table.loc[name,'PR(>F)']))
print(table)
# if table.loc['response:arealabelpair','PR(>F)'] < 0.05:
#     ax.text(0.4,0.85,'Interaction%s' % get_sig_asterisks(table.loc['response:arealabelpair','PR(>F)']),
#             fontsize=6,transform=ax.transAxes,ha='center',va='center')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_Modulation_vs_Activity_%dGRsessions' % (nSessions))


#%% Is the effect similar for the two areas, but 
# just dependent on a difference in activity levels?
# Is modulation more multiplicative for larger activity levels, stimuli with 
mincounts = 50
nbins = 10
ylims = [[-0.2,2],[-0.01,0.05]]

varstocontrol = ['gOSI','rangeresp','meanact']
ylabels = ['gOSI','response range','mean activity']

# fig,axes = plt.subplots(1,2*len(varstocontrol),figsize=(len(varstocontrol)*3.5*2*cm,3.5*cm),sharex=False,sharey=False)
fig,axes = plt.subplots(1,2*len(varstocontrol),figsize=(len(varstocontrol)*3.5*2*cm,3.5*cm),sharex=False,sharey=False)

for ivartocontrol,vartocontrol in enumerate(varstocontrol):
    for ivar,var in enumerate(['slope','offset']):
        df = pd.DataFrame({'controlvar':[],'vardata':[],'arealabel':[]})
        for ialp,alp in enumerate(arealabelpairs):
            ax = axes[ivartocontrol*2+ivar]

            # idx_N = celldata['arealayerlabel'] == alp.split('-')[2]

            idx_N = np.all((
                    celldata['arealayerlabel'] == alp.split('-')[2],
                    rangeresp>params['minrangeresp'],
                    np.any(dprimesig==1,axis=0),
                                                #  dprimesig[ialp,:]==sign

                    ),axis=0)
            
            bins = np.linspace(celldata[vartocontrol][idx_N].min(),np.nanpercentile(celldata[vartocontrol][idx_N],99),nbins)
            bincenters = (bins[1:]+bins[:-1])/2

            xdata = celldata[vartocontrol][idx_N]
            ydata = celldata[var][idx_N]

            idx_notnan = np.logical_and(~np.isnan(xdata),~np.isnan(ydata))
            xdata = xdata[idx_notnan]
            ydata = ydata[idx_notnan]
            df = pd.concat((df,pd.DataFrame({'controlvar':xdata,'vardata':ydata,'arealabel':ialp})))

            # ax = axes[ivar,0]
            ymeandata,_ = binned_statistic(xdata, ydata, statistic='mean', 
                                    bins=bins)[:2]
           
            bincounts = np.histogram(xdata,bins=bins)[0]
            yerror,_ = binned_statistic(xdata, ydata, statistic='std', 
                                    bins=bins)[:2]
            yerror /= np.sqrt(bincounts)

            ymeandata[bincounts<mincounts] = np.nan
            yerror[bincounts<mincounts] = np.nan
            # ax.plot(bin_edges[:-1],ymeandata,color=clrs_arealabelpairs[ialp],
                    # linewidth=2)
            shaded_error(bincenters,ymeandata,yerror,ax=ax,color=clrs_arealabelpairs[ialp])
            bincenters = bincenters[bincounts>=mincounts]
            # ax.scatter(xdata,ydata,s=0.5,alpha=0.5,color=clrs_arealabelpairs[ialp])
            ax.set_xlabel(ylabels[ivartocontrol])
            # ax.set_ylabel(var)
            ax.set_title(var,fontsize=8)
            ax.set_ylim(ylims[ivar])
            ax.set_xlim(bincenters[0],bincenters[-1])
        # df.dropna(inplace=True)

        # Perform the ANCOVA
        model = ols('vardata ~ arealabel + controlvar', data=df,hasconst=False).fit()
        # model = ols('vardata ~ arealabel', data=df).fit()
        # results = ols('vardata ~ arealabel + controlvar', data=df,hasconst=False).fit()
        # Print the summary of the model
        summary_table = model.summary().tables[0]
        # print(f"F={statistic}, p={p_value}")  # print(f"t-statistic: {statistic}")        # # Print the F-test statistic and P-value
        # Print the summary of the model
        summary_table = model.summary().tables[1]
        # print(summary_table)
        print('%s, %s: t=%1.1f, p=%1.2e' % (ylabels[ivartocontrol],var,model.tvalues[1],model.pvalues[1]))
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'ControlVars_AffineCoefs_%dGRsessions' % (nSessions))

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
                            #  celldata['gOSI']>0.4,
                             rangeresp>params['minrangeresp'],
                             dprimesig[ialp,:]==sign

                             ),axis=0)
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
            b = stats.linregress(xdata,ydata)
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
            # celldata['gOSI']>0,
            rangeresp>params['minrangeresp'],
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

