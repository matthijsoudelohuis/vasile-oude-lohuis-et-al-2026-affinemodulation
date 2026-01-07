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
# os.chdir('c:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')
os.chdir('c:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance,value_matching,filter_nearlabeled
from utils.plot_lib import * #get all the fixed color schemes
from loaddata.session_info import assign_layer,assign_layer2
from utils.RRRlib import regress_out_behavior_modulation

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\ControlAL\\')

# #%% 
# arealabelpairs  = [
#                     'V1lab-V1unl-PMunlL2/3',
#                     'PMlab-PMunl-V1unlL2/3',
#                     ]

# clrs_arealabelpairs         = get_clr_arealabelpairs(arealabelpairs)
# clrs_arealabels_low_high    = get_clr_area_low_high()  # PMlab-PMunl-V1unl

#%% Criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 0.5
minrangeresp            = 0.04

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


   #    #           #####  ####### #     # ####### ######  ####### #       
  # #   #          #     # #     # ##    #    #    #     # #     # #       
 #   #  #          #       #     # # #   #    #    #     # #     # #       
#     # #          #       #     # #  #  #    #    ######  #     # #       
####### #          #       #     # #   # #    #    #   #   #     # #       
#     # #          #     # #     # #    ##    #    #    #  #     # #       
#     # #######     #####  ####### #     #    #    #     # ####### ####### 


#%% Show tuning curve when activityin the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    'V1lab-V1unl-ALunlL2/3',
                    'PMlab-PMunl-ALunlL2/3',
                    # 'ALunl-V1unl-PMunlL2/3',
                    # 'ALunl-PMunl-V1unlL2/3',
                    ]


#%% Show tuning curve when activity in the other area is low or high (only still trials)
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
nboots                  = 200
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
                                    sessions[ises].celldata['nearby']
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                    sessions[ises].celldata['nearby']
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


# # %%
# celldata['meanact']     = np.nanmean(mean_resp_split,axis=(0,1,2))
# celldata['slope']       = np.nanmean(params_regress[:,:,0],axis=1)
# celldata['offset']      = np.nanmean(params_regress[:,:,1],axis=1)
# celldata['affine_R2']   = np.nanmean(params_regress[:,:,2],axis=1)
# celldata['rangeresp']   = rangeresp

#%% 
nresamples = 100
clrs_arealabelpairs = ['green','purple','orange','red','blue','grey']
legendlabels = ['$V1_{PM}-V1_{ND}>PM$',
                '$PM_{V1}-PM_{ND}>V1$',
                '$V1_{PM}-V1_{ND}>AL$',
                '$PM_{V1}-PM_{ND}>AL$',
                ]
narealabelpairs = len(arealabelpairs)

# clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(1,2,figsize=(6,3))
# fig,axes = plt.subplots(1,2,figsize=(12,6))
for iparam in range(2):
    ax = axes[iparam]
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        bins = np.arange(0,2,0.015)
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        bins = np.arange(-0.025,0.04,0.0001)
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    
    handles = []

    idx_N =  rangeresp>minrangeresp
    for ialp,alp in enumerate(arealabelpairs):
        if ialp<2:
            continue
        
        # print(np.sum(~np.isnan(params_regress[idx_N,ialp,iparam])))
        sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],
                     alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
        handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
    ax.legend(handles,legendlabels[2:],fontsize=9,frameon=False)

    #Test for effect in AL:
    h,p = stats.ttest_ind(params_regress[idx_N,2,iparam],
                            params_regress[idx_N,3,iparam],nan_policy='omit')
    p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
    ax.text(0.5, 0.5, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes)
    print('p=%1.2f' % (p))

    #Statistics:
    df = pd.DataFrame({'var': params_regress[idx_N,:,iparam].flatten(),
                       'arealabelpair': np.tile(arealabelpairs,np.sum(idx_N))})
    df.dropna(inplace=True)
    df['source'] = ''
    df.loc[df['arealabelpair'].str.contains('V1lab'),'source'] = 'V1lab'
    df.loc[df['arealabelpair'].str.contains('PMlab'),'source'] = 'PMlab'
    df['target'] = 'V1PM'
    df.loc[df['arealabelpair'].str.contains('AL'),'target'] = 'AL'

    formula = "var ~ source*target" #model with interaction
    lm = ols(formula, df).fit()
    table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
    print(table)
    print('Interaction effect: p=%1.2f' % (table.loc['source:target','PR(>F)']))

    #The number of recorded neurons in AL is smaller than those in V1 and PM:
    #pick nALneurons random entries from the entries that have 'target' set to 'V1PM' to match the number of target categories
    idx_V1PM = np.where(df['target'] == 'V1PM')[0] 
    idx_AL = np.where(df['target'] == 'AL')[0]
    # print(np.sum(df['target'] == 'AL'))
    # print(np.sum(df['target'] == 'V1PM'))
    fracsig = np.zeros(nresamples)
    for isub in range(nresamples):
        idx_V1PM_to_pick = np.random.choice(idx_V1PM, size=len(idx_AL), replace=False)

        df_sub = df.iloc[np.concatenate([idx_AL, idx_V1PM_to_pick]),:]

        lm = ols(formula, df_sub).fit()
        table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
        fracsig[isub] = table.loc['source:target','PR(>F)']<=0.05

    print('%2.1f%% of resamples had a significant interaction' % (np.sum(fracsig)/nresamples*100))
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'AL_HistCoefs_StillTrials_gOSI05_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])

#%%

nresamples = 10
narealabelpairs = len(arealabelpairs)

# clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(1,2,figsize=(6,3))
# fig,axes = plt.subplots(1,2,figsize=(12,6))
for iparam in range(2):
    ax = axes[iparam]
    handles = []

    idx_N =  rangeresp>minrangeresp
    # idx_N =  rangeresp
    # idx_N = celldata['gOSI']>0.4
    
    for ialp,alp in enumerate(arealabelpairs):
        ax.errorbar(ialp,np.nanmean(params_regress[idx_N,ialp,iparam]),
                    yerr=stats.sem(params_regress[idx_N,ialp,iparam],nan_policy='omit'),
                    color=clrs_arealabelpairs[ialp],
                    marker='o',markersize=8,linewidth=1.5,elinewidth=1.5,capsize=5)
    ax.legend(handles,legendlabels[2:],fontsize=9,frameon=False)

    #Test for effect in AL:
    h,p = stats.ttest_ind(params_regress[idx_N,2,iparam],
                            params_regress[idx_N,3,iparam],nan_policy='omit')
    p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
    ax.text(0.5, 0.5, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes)
    print('p=%1.2f' % (p))

    #Statistics:
    df = pd.DataFrame({'var': params_regress[idx_N,:,iparam].flatten(),
                       'arealabelpair': np.tile(arealabelpairs,np.sum(idx_N))})
    df.dropna(inplace=True)
    df['source'] = ''
    df.loc[df['arealabelpair'].str.contains('V1lab'),'source'] = 'V1lab'
    df.loc[df['arealabelpair'].str.contains('PMlab'),'source'] = 'PMlab'
    df['target'] = 'V1PM'
    df.loc[df['arealabelpair'].str.contains('AL'),'target'] = 'AL'

    formula = "var ~ source*target" #model with interaction
    lm = ols(formula, df).fit()
    table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
    print(table)
    print('Interaction effect: p=%1.2f' % (table.loc['source:target','PR(>F)']))

    #The number of recorded neurons in AL is smaller than those in V1 and PM:
    #pick nALneurons random entries from the entries that have 'target' set to 'V1PM' to match the number of target categories
    idx_V1PM = np.where(df['target'] == 'V1PM')[0] 
    idx_AL = np.where(df['target'] == 'AL')[0]
    # print(np.sum(df['target'] == 'AL'))
    # print(np.sum(df['target'] == 'V1PM'))
    fracsig = np.zeros(nresamples)
    for isub in range(nresamples):
        idx_V1PM_to_pick = np.random.choice(idx_V1PM, size=len(idx_AL), replace=False)

        df_sub = df.iloc[np.concatenate([idx_AL, idx_V1PM_to_pick]),:]

        lm = ols(formula, df_sub).fit()
        table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
        fracsig[isub] = table.loc['source:target','PR(>F)']<=0.05

    print('%2.1f%% of resamples had a significant interaction' % (np.sum(fracsig)/nresamples*100))
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'AL_HistCoefs_StillTrials_gOSI05_%dGRsessions' % (nSessions), formats = ['png'])
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])



#%%
ndiffs = 2
fracmat     = np.full((3,3,narealabelpairs+ndiffs),np.nan)
nsigmat     = np.full((3,3,narealabelpairs),np.nan)
ntotalmat   = np.full((3,3,narealabelpairs),np.nan)
testmat     = np.full((3,3),np.nan)
ncomparisons = 9
for ialp,alp in enumerate(arealabelpairs):
    # for imult, mult in enumerate([-1,0,1]):
    for imult, mult in enumerate([1,0,-1]):
        for iadd, add in enumerate([-1,0,1]):
            idx_N =  np.all((
                rangeresp>minrangeresp,
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
fracmat[:,:,4] = fracmat[:,:,2] - fracmat[:,:,0]
fracmat[:,:,5] = fracmat[:,:,3] - fracmat[:,:,1]

for imult, mult in enumerate([1,0,-1]):
    for iadd, add in enumerate([-1,0,1]):
        data = np.array([[nsigmat[imult,iadd,0], ntotalmat[imult,iadd,0]-nsigmat[imult,iadd,0]],
                         [nsigmat[imult,iadd,1], ntotalmat[imult,iadd,1]-nsigmat[imult,iadd,1]]])
        testmat[imult,iadd] = stats.chi2_contingency(data)[1]  # p-value
testmat = testmat * ncomparisons  #bonferroni correction

fig,axes = plt.subplots(1,narealabelpairs+ndiffs,figsize=((narealabelpairs+ndiffs)*3,3))
for ialp in range(narealabelpairs+ndiffs):
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
# my_savefig(fig,savedir,'Affine_sig_mod_FF_FB_heatmap_%dsessions' % (nSessions))


#%% 
# nresamples = 100
# clrs_arealabelpairs = ['green','purple','orange','red','blue','grey']
# legendlabels = ['$V1_{PM}>PM$',
#                 '$PM_{V1}>V1$',
#                 '$V1_{PM}>AL$',
#                 '$PM_{V1}>AL$',
#                 ]
# narealabelpairs = len(arealabelpairs)

# #pick nALneurons random entries from the entries that have 'target' set to 'V1PM' to match the number of target categories
# nALneurons = np.sum(df['target'] == 'AL')
# nALneurons = 200

# params_regress_mean = np.zeros((narealabelpairs,2))
# for isub in range(nresamples):
#     params_regress_diff = np.stack((params_regress[:,1,:]-params_regress[:,0,:],params_regress[:,3,:]+params_regress[:,2,:]),axis=1)

# # clrs_arealabelpairs = ['grey','pink','grey','red']
# fig,axes = plt.subplots(1,2,figsize=(6,3))
# # fig,axes = plt.subplots(1,2,figsize=(12,6))
# for iparam in range(2):
#     ax = axes[iparam]
#     if iparam == 0:
#         ax.set_xlabel('Multiplicative Slope')
#         bins = np.arange(-0.5,5,0.05)
#         ax.axvline(1,color='grey',ls='--',linewidth=1)
#     else:
#         ax.set_xlabel('Additive Offset')
#         bins = np.arange(-0.05,0.05,0.0005)
#         ax.axvline(0,color='grey',ls='--',linewidth=1)
#     handles = []

#     # idx_N =  celldata['gOSI']>0.4
#     idx_N =  rangeresp>0.04
#     for ialp,alp in enumerate(arealabelpairs):
#         if ialp<2:
#             continue
        
#         # print(np.sum(~np.isnan(params_regress[idx_N,ialp,iparam])))
#         sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
#                      color=clrs_arealabelpairs[ialp],
#                      alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
#         handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.2,markersize=10,
#                 color=clrs_arealabelpairs[ialp],marker='v')[0])
#     ax.legend(handles,legendlabels[2:],fontsize=9,frameon=False)

#     #Test for effect in AL:
#     h,p = stats.ttest_ind(params_regress[idx_N,2,iparam],
#                             params_regress[idx_N,3,iparam],nan_policy='omit')
#     p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
#     ax.text(0.5, 0.5, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes)
#     print('p=%1.2f' % (p))

#     #Statistics:
#     df = pd.DataFrame({'var': params_regress[idx_N,:,iparam].flatten(),
#                        'arealabelpair': np.tile(arealabelpairs,np.sum(idx_N))})
#     df.dropna(inplace=True)
#     df['source'] = ''
#     df.loc[df['arealabelpair'].str.contains('V1lab'),'source'] = 'V1lab'
#     df.loc[df['arealabelpair'].str.contains('PMlab'),'source'] = 'PMlab'
#     df['target'] = 'V1PM'
#     df.loc[df['arealabelpair'].str.contains('AL'),'target'] = 'AL'

#     formula = "var ~ source*target" #model with interaction
#     lm = ols(formula, df).fit()
#     table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
#     print(table)
#     print('Interaction effect: p=%1.2f' % (table.loc['source:target','PR(>F)']))

#     #The number of recorded neurons in AL is smaller than those in V1 and PM:
#     #pick nALneurons random entries from the entries that have 'target' set to 'V1PM' to match the number of target categories
#     idx_V1PM = np.where(df['target'] == 'V1PM')[0] 
#     idx_AL = np.where(df['target'] == 'AL')[0]
#     # print(np.sum(df['target'] == 'AL'))
#     # print(np.sum(df['target'] == 'V1PM'))
#     fracsig = np.zeros(nresamples)
#     for isub in range(nresamples):
#         idx_V1PM_to_pick = np.random.choice(idx_V1PM, size=len(idx_AL), replace=False)

#         df_sub = df.iloc[np.concatenate([idx_AL, idx_V1PM_to_pick]),:]

#         lm = ols(formula, df_sub).fit()
#         table = anova_lm(lm, typ=2) # Type 2 ANOVA DataFrame
#         fracsig[isub] = table.loc['source:target','PR(>F)']<=0.05

#     print('%2.1f%% of resamples had a significant interaction' % (np.sum(fracsig)/nresamples*100))
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'AL_HistCoefs_StillTrials_gOSI05_%dGRsessions' % (nSessions), formats = ['png'])
# # my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])
