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
from scipy.signal import detrend
from scipy.optimize import curve_fit

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

#%%
######  ######  ####### #######     #####  ####### ### #     # 
#     # #     # #       #          #     #    #     #  ##   ## 
#     # #     # #       #          #          #     #  # # # # 
######  ######  #####   #####       #####     #     #  #  #  # 
#       #   #   #       #                #    #     #  #     # 
#       #    #  #       #          #     #    #     #  #     # 
#       #     # ####### #           #####     #    ### #     # 

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
# my_savefig(fig,savedir,'PrefStim_%dGRsessions' % (nSessions))


#%%

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
######     #    ####### #######    #     # ####### ####### ######  ###  #####  
#     #   # #      #    #          ##   ## #          #    #     #  #  #     # 
#     #  #   #     #    #          # # # # #          #    #     #  #  #       
######  #     #    #    #####      #  #  # #####      #    ######   #  #       
#   #   #######    #    #          #     # #          #    #   #    #  #       
#    #  #     #    #    #          #     # #          #    #    #   #  #     # 
#     # #     #    #    #######    #     # #######    #    #     # ###  #####  

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
fig, axes = plt.subplots(1,1,figsize=(6*cm,5*cm))
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

print('r=%1.2f+-%1.2f, (%d sessions) p=%1.2e, one sample t-test' % (np.nanmean(tempdata),np.nanstd(tempdata),nSessions,stats.ttest_1samp(tempdata,0)[1]))

#%% 
ises = np.where(sessiondata['session_id']=='LPE09665_2023_03_14')[0][0]

respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']
legendlabels    = ['FF','FB']

np.random.seed(1)
trialstoplot = np.arange(1020,1110)

clrs = ['red','grey']
fig,axes = plt.subplots(2,2,figsize=(10*cm,5*cm),sharex=True,sharey=False)
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[0,ialp]

    idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]
    idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]

    meanpopact_N1       = np.nanmean(respdata[idx_source_N1,:],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)
    meanpopact_N2       = np.nanmean(respdata[idx_source_N2,:],axis=0)
    meanpopact_N3       = meanpopact_N1 - meanpopact_N2
    meanpopact_N3 = detrend(meanpopact_N3)

    ax.plot(meanpopact_N1,color=clrs[0],marker='',markersize=5,linewidth=1)
    ax.plot(meanpopact_N2,color=clrs[1],marker='',markersize=5,linewidth=1)

    # ax.plot(meanpopact_N3,color='grey',linewidth=0.4)
    ax.legend(alp.split('-'),frameon=False)
    my_legend_strip(ax)
    ax.set_xlim(np.percentile(trialstoplot,[0,100]))
    ax.set_ylim(np.percentile([meanpopact_N1[trialstoplot],meanpopact_N2[trialstoplot]],[0,100]))
    ax.set_title(legendlabels[ialp])
    if ialp==0:
        ax.set_ylabel('avg activity (ev/F0)')
    ax.text(0.1,0.8,'r = %1.2f, p = %1.2f' % (stats.pearsonr(meanpopact_N1,meanpopact_N2)),transform=ax.transAxes,fontsize=6)
    ax_nticks(ax,4)

    ax = axes[1,ialp]
    # ax.plot(meanpopact_N1-meanpopact_N2,color=clrs[0],marker='.',markersize=5,linewidth=1)
    ax.fill_between(np.arange(len(meanpopact_N3)),meanpopact_N3,color='k',linewidth=0.8)
    # ax.set_xlim(np.percentile(trialstoplot,[0,100]))
    ax.set_ylim(np.percentile([meanpopact_N3[trialstoplot],meanpopact_N3[trialstoplot]],[0,100]))
    ax.set_xlabel('trial')
    if ialp==0:
        ax.set_ylabel('activity diff')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=5,trim=False)
# my_savefig(fig,savedir,'PopulationActivity_Trials_FF_FB_%d-%d_%s' % (trialstoplot[0]+1,trialstoplot[-1]+1,sessions[ises].session_id))

#%% Compute autocorrelation function
nlagtrials = 75
lags = np.arange(-nlagtrials, nlagtrials + 1)

# ac_data = np.full((nSessions,narealabelpairs,2*nlagtrials+1),np.nan)
ac_data = np.full((2,nSessions,narealabelpairs,2*nlagtrials+1),np.nan)
clrs = ['red', 'grey']
np.random.seed(9)
for ises in range(nSessions):  
    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    for icontrol,control in enumerate([False,True]):
        arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl'] if not control else ['V1unl-V1unl','PMunl-PMunl']

        for ialp,alp in enumerate(arealabelpairs):
            idx_source_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                        sessions[ises].celldata['nearby']
                                                        ),axis=0))[0]
            idx_source_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                        sessions[ises].celldata['nearby']
                                                        ),axis=0))[0]
            if (len(idx_source_N1) < params['minnneurons']) or (len(idx_source_N2) < params['minnneurons']):
                continue

            if control:
                subsampleneurons    = len(idx_source_N1)//2
                idx_source_N1       = np.random.choice(idx_source_N1,subsampleneurons,replace=False)
                idx_source_N2       = np.random.choice(np.setdiff1d(idx_source_N2,idx_source_N1),subsampleneurons,replace=False)

            meanpopact_N1       = np.nanmean(respdata[idx_source_N1,:],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)
            meanpopact_N2       = np.nanmean(respdata[idx_source_N2,:],axis=0)
            meanpopact_N3       = meanpopact_N1 - meanpopact_N2

            # Compute autocorrelation of meanpopdiff over 10 trials
            x = meanpopact_N3 - np.nanmean(meanpopact_N3)
            x = detrend(x)
            # replace nan with zero for correlation
            x = np.nan_to_num(x)
            ac_full = np.correlate(x, x, mode='full')
            center = len(ac_full) // 2
            ac_segment = ac_full[center - nlagtrials:center + nlagtrials + 1]
            # normalize
            ac_segment = ac_segment / ac_full[center]
            ac_data[icontrol,ises,ialp,:] = ac_segment

ac_data[:,:,:,lags==0] = np.nan

# ac_data = smooth
fig,axes = plt.subplots(1,2,figsize=(6.2*cm,3*cm))
for ialp,alp in enumerate(arealabelpairs):
    ax=axes[ialp]
    for icontrol in range(2):
        shaded_error(lags,ac_data[icontrol,:,ialp,:],error='sem',color=clrs[icontrol],ax=ax,linewidth=0.5)
        # ax.plot(lags,np.nanmean(ac_data[icontrol,:,ialp,:],axis=0),marker='.',color=clrs[icontrol],markersize=1,linewidth=0)

        # Fit exponential decay to autocorrelation
        ac_mean = np.nanmean(ac_data[icontrol,:,ialp,:],axis=0)
        lags_pos = lags[lags>0]
        ac_pos = ac_mean[lags>0]

        # Fit exponential: A * exp(-t/tau)
        def exp_decay(t, A, tau):
            return A * np.exp(-np.abs(t) / tau)
        
        try:
            popt, _ = curve_fit(exp_decay, lags_pos, ac_pos, p0=[0.1, 5], maxfev=5000)
            tau = popt[1]
            ax.text(0.98, 0.97-0.1*icontrol, r'$\tau$ = %.2f' % tau, transform=ax.transAxes, 
                    ha='right', va='top', fontsize=6,color=clrs[icontrol])
            # Plot fitted exponential curve
            lags_fit = np.linspace(0, nlagtrials, 100)
            ac_fit = exp_decay(lags_fit, popt[0], popt[1])
            # ax.plot(lags_fit, ac_fit, '-', color=clrs[icontrol],linewidth=1, alpha=0.7, label='Exponential fit')
        except:
            pass
    ax_nticks(ax,4)
    ax.set_title(legendlabels[ialp])
    ax.axhline(0,linewidth=0.8,linestyle=':',color='black')
    ax.set_xlim([0,nlagtrials])
    ax.set_ylim([-0.01,0.22])
    ax.set_xlabel('lag (trials)')
    ax.set_ylabel('autocorrelation')
sns.despine(fig=fig, top=True, right=True,offset=2)
# my_savefig(fig,savedir,'Autocorrelation_diffrate_%dsessions_line' % (nSessions))
# my_savefig(fig,savedir,'Autocorrelation_diffrate_%dsessions_fit' % (nSessions))


#%% 
ises = np.where(sessiondata['session_id']=='LPE10919_2023_11_06')[0][0]
vscale      = 0.015
markersize = 1.3
markeralpha = 0.7
np.random.seed(6)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['blue','grey','red'])
idx_T_still = np.where(np.all((
                        sessions[ises].respmat_videome < params['maxvideome'],
                        sessions[ises].respmat_runspeed < params['maxrunspeed'],
                        ),axis=0))[0]
print('n=%d trials' % len(idx_T_still))
ustims = np.unique(sessions[ises].trialdata['Orientation'])
nstim = len(ustims)

fig,axes = plt.subplots(2,1,figsize=(3.7*cm,6.3*cm))
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

    meanpopact_N1       = np.nanmean(respdata[idx_source_N1,:],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)
    meanpopact_N2       = np.nanmean(respdata[idx_source_N2,:],axis=0)
    meanpopact_diff     = meanpopact_N1 - meanpopact_N2

    vmin,vmax = np.percentile(meanpopact_diff[idx_T_still],[5,95])
    absmax = my_ceil(np.max(np.abs([vmin,vmax])),2)
    vmin,vmax = -absmax,absmax

    ax=axes[ialp]
    # handle = ax.scatter(meanpopact_N2[idx_T_still],meanpopact_N1[idx_T_still],c=meanpopact_diff[[idx_T_still]],edgecolor='none',
                        # s=2,cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.8)
    idx_high = np.array([]).astype(int)
    idx_low = np.array([]).astype(int)
    for istim,stim in enumerate(ustims):
        idx_T = np.all((sessions[ises].respmat_videome < params['maxvideome'],
                                sessions[ises].respmat_runspeed < params['maxrunspeed'],
                                sessions[ises].trialdata['Orientation'] == stim),axis=0)

        idx_K1              = meanpopact_diff < np.nanpercentile(meanpopact_diff[idx_T],params['splitperc'])
        idx_K2              = meanpopact_diff > np.nanpercentile(meanpopact_diff[idx_T],100-params['splitperc'])
        idx_high            = np.concatenate((idx_high,np.where(np.logical_and(idx_T,idx_K2))[0]))
        idx_low             = np.concatenate((idx_low,np.where(np.logical_and(idx_T,idx_K1))[0]))
    idx_rest = np.setdiff1d(idx_T_still,[idx_high,idx_low])
    handle = ax.scatter(meanpopact_N2[idx_rest],meanpopact_N1[idx_rest],color='grey',edgecolor='none',
                        s=markersize,alpha=markeralpha)
    handle = ax.scatter(meanpopact_N2[idx_low],meanpopact_N1[idx_low],color=clrs_arealabels_low_high[ialp,0],edgecolor='none',
                        s=markersize,alpha=markeralpha)
    handle = ax.scatter(meanpopact_N2[idx_high],meanpopact_N1[idx_high],color=clrs_arealabels_low_high[ialp,1],edgecolor='none',
                        s=markersize,alpha=markeralpha)
    # meanpopact          = meanpopact_N1 - meanpopact_N2
    # topthr              = np.percentile(meanpopact,100-params['splitperc'])
    # bottomthr           = np.percentile(meanpopact,params['splitperc'])

    ax.plot([-1,1],[-1,1],color='k',linewidth=0.5,linestyle='--')
    ax.set_xlim(np.percentile([meanpopact_N2[idx_T_still],meanpopact_N1[idx_T_still]],[0.1,99.9]))
    ax.set_ylim(np.percentile([meanpopact_N2[idx_T_still],meanpopact_N1[idx_T_still]],[0.1,99.9]))

    ax.set_xlabel('%s (events/F0)' % arealabeled_to_figlabels([alp.split('-')[1]]),color='k')
    ax.set_ylabel('%s (events/F0)' % arealabeled_to_figlabels([alp.split('-')[0]]), color='k')
    ax_nticks(ax, 3)
    ax.set_xticks([0,0.05,0.1])
    ax.set_yticks([0,0.05,0.1])
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=7)
    # cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))

sns.despine(fig=fig, top=True, right=True)
my_savefig(fig,savedir,'Diff_LabUnl_%s' % sessiondata['session_id'][ises])

#%% Show the distribution of mean activity of the labeled population 
# and the selection of the 25% lowest and 25% highest activity trials:
arealabelpairs  = ['V1lab','PMlab']
arealabelpairs  = ['V1lab-V1unl','PMlab-PMunl']
titlelabels     = ['Feedforward','Feedback']
legendlabels    = ['FF','FB']

respdata        = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

clrs = ['black','grey','red']

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

    bins = allbins[np.logical_and(allbins >= np.percentile(meanpopact[idx_T_still],params['splitperc'])-step, allbins <= np.percentile(meanpopact[idx_T_still],100-params['splitperc']))]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color='grey',edgecolor='none')

    bins = allbins[allbins > np.percentile(meanpopact[idx_T_still]-step,100-params['splitperc'])]
    sns.histplot(data=meanpopact[idx_T_still],ax=ax,kde=False,bins = bins,color=clrs_arealabels_low_high[ial,1],edgecolor='none')


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

        sns.histplot(data=meanpopact[idx_T],ax=ax,kde=False,bins = bins,color=clrs_arealabels_low_high[ial,0],edgecolor='none')
        bins = allbins[np.logical_and(allbins >= np.percentile(meanpopact[idx_T],params['splitperc']), 
                                      allbins <= np.percentile(meanpopact[idx_T],100-params['splitperc'])+step)]
        sns.histplot(data=meanpopact[idx_T],ax=ax,kde=False,bins = bins,color='grey',edgecolor='none')
        bins = allbins[allbins > np.percentile(meanpopact[idx_T],100-params['splitperc'])]
        sns.histplot(data=meanpopact[idx_T],ax=ax,kde=False,bins = bins,color=clrs_arealabels_low_high[ial,1],edgecolor='none')

        ax.set_xlim(allbins[0],allbins[-1])
        ax.get_yaxis().set_visible(False)
        ax.text(0.95,0.7,stim,fontsize=4,ha='right',va='bottom',transform=ax.transAxes)
        ax.set_xlabel('%s (events/F0)' % arealabelpair_to_figlabel(alp)[0])

        # ax.plot([0,0],[0,1],color='k',lw=1,transform=ax.get_yaxis_transform())
sns.despine(fig=fig, top=True, right=True, left=True,offset=2,trim=False)
# my_savefig(fig,savedir,'Hist_PopAct_FF_FB_perStim_%s' % sessiondata['session_id'][ises])


#%% Correlate difference in activity metrics across sessions:
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

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

    diffdata            = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    targetareadata      = np.full((narealabelpairs,np.sum(idx_T_still)),np.nan)
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                        sessions[ises].celldata['nearby']
                                                        ),axis=0))[0]
        idx_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                        sessions[ises].celldata['nearby']
                                                        ),axis=0))[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

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

print('The difference metrics for FF and FB (V1pm-V1nd with PMv1-PMnd) were not correlated (r = %1.2f +- %1.2f across n=%d sessions with both FF and FB, p = %1.2f one sample t-test)' % (np.nanmean(corrdata_labdiff_ses),
                                                                            np.nanstd(corrdata_labdiff_ses),np.sum(~np.any(np.isnan(corrdata_targetarea_ses),axis=0)),
                                                                            stats.ttest_1samp(corrdata_labdiff_ses,0,nan_policy='omit').pvalue))
for ialp in range(narealabelpairs):
    print('%s: r = %1.2f (std = %1.2f), p = %1.2f' % (arealabelpairs[ialp],np.nanmean(corrdata_targetarea_ses[ialp,:]),
                                                     np.nanstd(corrdata_targetarea_ses[ialp,:]),stats.ttest_1samp(corrdata_targetarea_ses[ialp,:],0,nan_policy='omit').pvalue))

#%% Is the difference metric consistent between different subsamples of neurons? 
sampleneurons   = np.array([1,10,20,30,40,50,60,70])
nsampleneurons  = len(sampleneurons)
nresamples      = 75
corrdata        = np.full((nSessions,2,nsampleneurons,nresamples),np.nan)
nlabeled_neurons = np.full((nSessions,2),np.nan)
for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                    sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]
        idx_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                    sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]
        nlabeled_neurons[ises,ialp] = len(idx_N1)
        for iNN,N in enumerate(sampleneurons):
            if len(idx_N1) < N*2 or len(idx_N2) < N*2:
                continue

            meanpopact_N2       = np.nanmean(respdata[np.ix_(idx_N2,idx_T_still)],axis=0)

            for isample in range(nresamples):
                idx_source_N1_1     = np.random.choice(idx_N1,N,replace=False)
                idx_source_N1_2     = np.random.choice(idx_N1,N,replace=False)

                meanpopact_N1_1       = np.nanmean(respdata[np.ix_(idx_source_N1_1,idx_T_still)],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)
                meanpopact_N1_2       = np.nanmean(respdata[np.ix_(idx_source_N1_2,idx_T_still)],axis=0) #np.nanmean(respdata[idx_source_N1,:],axis=0)

                diffpopact_1       = meanpopact_N1_1 - meanpopact_N2
                diffpopact_2       = meanpopact_N1_2 - meanpopact_N2

                corrdata[ises,ialp,iNN,isample] = stats.pearsonr(diffpopact_1,diffpopact_2)[0]

#%% 
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.5*cm))
for ialp,alp in enumerate(arealabelpairs):
    ax = axes
    ax.plot(sampleneurons,np.nanmean(corrdata[:,ialp,:,:],axis=(0,2)),'-',marker='o',markersize=2,
            color=clrs_arealabelpairs[ialp],linewidth=1)
    shaded_error(sampleneurons,np.nanmean(corrdata[:,ialp,:,:].T,axis=(2)),center='mean',error='std',
            color=clrs_arealabelpairs[ialp],linewidth=1,ax=ax)
    # for ises in range(nSessions):
        # if nlabeled_neurons[ises,ialp] > params['minnneurons']:
            # ax.axvline(nlabeled_neurons[ises,ialp],color=clrs_arealabelpairs[ialp],linestyle='-',linewidth=0.3)
    nlabeled_neurons[nlabeled_neurons<params['minnneurons']] = np.nan
    ax.axvline(np.nanmean(nlabeled_neurons[:,ialp]),color=clrs_arealabelpairs[ialp],linestyle='-',lw=0.5)
    ax.set_xlabel('#neurons sampled')
    ax.set_ylabel('corr between subsamples')
    ax.set_xticks(sampleneurons)
    ax.set_xticks(sampleneurons)
    ax.set_ylim([0,0.7])
    ax.set_yticks([0,0.2,0.4,0.6])

sns.despine(fig=fig, top=True, right=True,offset=2)
# my_savefig(fig,savedir,'InternalConsistency_SubsampleCorr_FF_FB_acrosssessions')
# my_savefig(fig,savedir,'InternalConsistency_SubsampleCorr_FF_FB_acrosssessions_wnNeurons')

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

# ndprimeboots            = 250
# ndprimeboots            = 1000
ndprimeboots            = 0
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

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response
        mean_resp_split_aligned[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

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

#%% Compute same metric as Flora and report n cells included/excluded based on this:
rangeresp2 = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp2 = np.nanmax(rangeresp2,axis=(1))

for ialp in range(narealabelpairs):
    print('%s: %d/%d cells included for affine modulation based on min range resp of %1.2f:' % (arealabelpairs[ialp],
            np.sum(rangeresp2[ialp,:]>params['minrangeresp']),np.sum(~np.isnan(rangeresp2[ialp,:])),params['minrangeresp']))

#%% Show the activity across trials of example cells that are positively and negatively modulated by 
# the high and low feedforward activity, respectively:
ises = 0

#%% Show an example plane: 
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
for ises in range(nSessions):
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    sessions[ises].celldata['dprime']   = np.nanmean(dprimedata[:,idx_ses],axis=0)
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%%
cellfield = 'dprime'
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

#%% 

   #    ####### ####### ### #     # #######     #####  #     #    #    ######     #     #####  
  # #   #       #        #  ##    # #          #     # #     #   # #   #     #   # #   #     # 
 #   #  #       #        #  # #   # #          #       #     #  #   #  #     #  #   #  #       
#     # #####   #####    #  #  #  # #####      #       ####### #     # ######  #     # #       
####### #       #        #  #   # # #          #       #     # ####### #   #   ####### #       
#     # #       #        #  #    ## #          #     # #     # #     # #    #  #     # #     # 
#     # #       #       ### #     # #######     #####  #     # #     # #     # #     #  #####  

#%% Show some example neurons:

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

#%% Evaluating potential confound of anatomical proximity:
# To evaluate potential confounds due to anatomical inhomogeneities in modulations, we first measured the absolute difference in 
# modulation as a function of distance between pairs of cells, regardless of their projection target. We found that cells that are
# in close anatomical proximity are modulated more similarly (Fig. S10a,b).

distancemetric = 'xy'
step                = 20
binedges            = np.arange(0,300 + step,step)
bincenters          = binedges[:-1] + step/2
nbins              = len(bincenters)
deltadprime         = np.full((2,nbins,nSessions),np.nan) #store absolute difference in modulation between pairs of cells for each session
# deltadprime         = np.full((2,nradii,nSessions),np.nan) #store absolute difference in modulation between pairs of cells for each session

for ises in tqdm(range(nSessions),desc='Computing modulation difference as a function of distance'):
    idx_ses = np.isin(celldata['session_id'],sessions[ises].session_id)
    #get pairs of cells that are within this radius of each other:
    if distancemetric == 'xyz': 
        distmat = sessions[ises].distmat_xyz
    elif distancemetric == 'xy':
        distmat = sessions[ises].distmat_xy
    deltadprime_FF = np.abs(dprimedata[0,idx_ses][:,None] - dprimedata[0,idx_ses][None,:])
    deltadprime_FB = np.abs(dprimedata[1,idx_ses][:,None] - dprimedata[1,idx_ses][None,:])
    for ibin,binedge in enumerate(binedges[:-1]):
        idx_pairs = np.logical_and(distmat>binedges[ibin], distmat<=binedges[ibin+1])
        # idx_pairs = np.where(np.triu(np.logical_and(distmat>radii[irad], distmat<=radii[irad+1]),1))[0]
        # cellpairs = np.array(np.triu_indices(np.sum(idx_ses),1)).T[idx_pairs]
        deltadprime[0,ibin,ises] = np.nanmean(deltadprime_FF[idx_pairs])
        deltadprime[1,ibin,ises] = np.nanmean(deltadprime_FB[idx_pairs])

#%% Show the delta prime as a function of distance between pairs of cells:
fig,ax = plt.subplots(1,1,figsize=(4.3*cm,3.8*cm),sharex=True,sharey=True)
handles = []
for ialp,alp in enumerate(legendlabels):
    handles.append(shaded_error(bincenters,deltadprime[ialp,:,:].T,error='sem',ax=ax,color=clrs_arealabelpairs[ialp],alpha=0.25,linewidth=1.5))
    # ax.plot(bincenters,deltadprime[ialp,:,:],color=clrs_arealabelpairs[ialp],linewidth=0.5)

ax.set_xticks(np.arange(0,binedges[-1]+10,25))
ax.set_xlabel(r'Distance between cell pairs ($\mu$m)')
ax_nticks(ax,5)
ax.set_xlim([0,binedges[-1]])
ax.set_ylabel(r'|$\Delta$ dprime|')
ax.legend(handles,legendlabels,fontsize=6,loc='lower right',reverse=True)
my_legend_strip(ax)

for ialp,alp in enumerate(legendlabels):
    df = nd_array_to_dataframe(deltadprime[ialp],dim_names=['distance', 'session'])

    #statistics: 
    formula = "value ~ distance + C(session)" #include session as random effect to account for repeated measures across sessions
    lm = ols(formula, df).fit()
    table = anova_lm(lm, typ=2) # Type 2 ANOVA
    for name in ['distance']:
        print('%s: F=%2.1f,p=%2.3g' % (legendlabels[ialp],table.loc[name,'F'],table.loc[name,'PR(>F)']))

sns.despine(fig=fig, top=True, right=True,offset=2)
# my_savefig(fig,savedir,'Delta_dprime_distance_V1PM_%dsessions' % nSessions)
