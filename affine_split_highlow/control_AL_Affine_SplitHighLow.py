#%% 
import os
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

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.session_info import *
from loaddata.get_data_folder import get_local_drive
from utils.pair_lib import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *
from utils.tuning import *
from utils.gain_lib import *

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\ControlAL\\')

#%% Plotting and parameters:
params  = load_params()
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_noiselevel=True)

#%%  Load data properly:
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=params['calciumversion'])
    # sessions[ises].respmat  /= sessions[ises].celldata['meanF'].to_numpy()[:,None] #convert to deconv/F0

#%% Compute Tuning Metrics (gOSI, gDSI etc.)
sessions = ori_remapping(sessions)
sessions = compute_tuning_wrapper(sessions)

#%% Compute nearby cells (for subsampling in some analyses to control for spatial clustering of recorded neurons)
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'])

#%% Get concatenated data:
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)


#%%
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
# #%% Get colors
# arealabelpairs  = ['V1lab-V1unl-PMunlL2/3',
#                     'PMlab-PMunl-V1unlL2/3']
# clrs_arealabelpairs         = get_clr_arealabelpairs(arealabelpairs)
# clrs_arealabels_low_high    = get_clr_area_low_high()  # PMlab-PMunl-V1unl
# clrs_labeled = get_clr_labeled(['unl','lab'])



#%% Show tuning curve when activity in the other area is low or high (only still trials)
narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())

mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells),np.nan)

#Regression output:
nboots                  = 0
# nboots                  = 250
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

#Dprime output:
ndprimeboots            = 250
# ndprimeboots = 0
dprimedata              = np.full((narealabelpairs,nCells),np.nan)
dprimesig              = np.full((narealabelpairs,nCells),np.nan)

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

            bootregress_sig[regressdata[:,0]>np.percentile(bootregressdata[:,:,0],97.5,axis=1),0] = 1
            bootregress_sig[regressdata[:,0]<np.percentile(bootregressdata[:,:,0],2.5,axis=1),0] = -1
            bootregress_sig[regressdata[:,1]>np.percentile(bootregressdata[:,:,1],97.5,axis=1),1] = 1
            bootregress_sig[regressdata[:,1]<np.percentile(bootregressdata[:,:,1],2.5,axis=1),1] = -1

            sig_params_regress[idx_ses,ialp,:] = bootregress_sig[idx_N3]

        #Aligned:
        prefori                     = np.argmax(np.mean(meanresp,axis=2),axis=1)

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response
        # tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        # meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        # meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

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

        if ndprimeboots:
            bootdprimedata  = np.full((N,ndprimeboots),np.nan)
            bootdprime_sig  = np.full((N),0)
            for iboot in range(ndprimeboots):
                idx_K1              = np.random.choice(np.where(idx_T_still)[0],size=np.sum(idx_T_still)*params['splitperc']//100,replace=False)
                idx_K2              = np.random.choice(np.where(idx_T_still)[0],size=np.sum(idx_T_still)*params['splitperc']//100,replace=False)
               
                bootdprimedata[:,iboot] = compute_dprime_mat(respdata[:,idx_K2],respdata[:,idx_K1])

            bootdprime_sig[dprime_ses>np.percentile(bootdprimedata,100-(params['dprime_alpha']/2*100),axis=1)] = 1
            bootdprime_sig[dprime_ses<np.percentile(bootdprimedata,params['dprime_alpha']/2*100,axis=1)] = -1

            dprimesig[ialp,idx_ses] = bootdprime_sig[idx_N3]
        
# Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

#%% Show pie chart of significant correlation for feedforward and feedback:
sigmat = np.empty((3,narealabelpairs))
# signorder = [-1,0,1]
# signlabels = ['Neg','None','Pos']
signorder = [-1,1,0]
signlabels = ['Neg','Pos','None']   
clrs_signs = ['#0033B3','#E66804','#808080']
legendlabels        = ['FF','FB']
targetarealabels    = ['PM','V1']

legendlabels        = ['V1-PM','PM-V1','V1-AL','PM-AL']
targetarealabels    = ['PM','V1','AL','AL']

for ialp,alp in enumerate(arealabelpairs):
    for isign,sign in enumerate(signorder):
        # idx_N = np.logical_and(rangeresp>params['minrangeresp'],~np.isnan(corrdata_cells[ialp,:]))
        idx_N = ~np.isnan(dprimesig[ialp,:])
        sigmat[isign,ialp] = np.sum(dprimesig[ialp,idx_N]==sign) / np.sum(idx_N) / (1+np.abs(sign)*(ialp//2))

#Make the figure:
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3*cm,3.5*cm))
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    ax.pie([sigmat[0,ialp],sigmat[1,ialp],sigmat[2,ialp]],labels=signlabels,colors=clrs_signs,autopct='%1.1f%%',
            startangle=90,counterclock=False,wedgeprops = {'linewidth': 0.8, 'edgecolor': 'black', 'alpha': 0.7},
            textprops={'fontsize': 7})
    ax.set_title('%s\nn=%d' % (legendlabels[ialp],np.sum(~np.isnan(dprimesig[ialp,:]))))
# my_savefig(fig,savedir,'Dprime_Sign_PieCharts_%dsessions_controlAL' % nSessions)



#%% Deprecated: 


#%% 
nresamples = 1
clrs_arealabelpairs = ['green','purple','orange','red','blue','grey']
legendlabels = ['$V1_{PM}-V1_{ND}>PM$',
                '$PM_{V1}-PM_{ND}>V1$',
                '$V1_{PM}-V1_{ND}>AL$',
                '$PM_{V1}-PM_{ND}>AL$',
                ]
narealabelpairs = len(arealabelpairs)

# clrs_arealabelpairs = ['grey','pink','grey','red']
fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm))
# fig,axes = plt.subplots(1,2,figsize=(12,6))
for iparam in range(2):
    ax = axes[iparam]
    if iparam == 0:
        ax.set_xlabel('Multiplicative Slope')
        # bins = np.arange(0,2,0.015)
        bins = np.arange(-0.5,5,0.015)
        ax.axvline(1,color='grey',ls='--',linewidth=1)
    else:
        ax.set_xlabel('Additive Offset')
        # bins = np.arange(-0.025,0.04,0.0001)
        bins = np.arange(-0.05,0.1,0.0001)
        ax.axvline(0,color='grey',ls='--',linewidth=1)
    
    handles = []

    idx_N = rangeresp>params['minrangeresp']
    # idx_N = np.all((
            # rangeresp>minrangeresp,
            # # celldata['noise_level']<maxnoiselevel,
            # ),axis=0)
    
    for ialp,alp in enumerate(arealabelpairs):
        if ialp<2:
            continue
        
        sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],
                     alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
        handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.7+ialp*0.1,markersize=6,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
    ax.legend(handles,legendlabels[2:],frameon=False)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'AL_HistCoefs_StillTrials_%dGRsessions' % (nSessions))
# my_savefig(fig,savedir,'FF_FB_affinemodulation_StillTrials_gOSI05_cumhistcoefs_%dGRsessions' % (nSessions), formats = ['png'])

#%% 
narealabelpairs = len(arealabelpairs)
ylims = np.array([[0.8,1.5],[0,0.013]])
fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm))
for iparam in range(2):
    ax = axes[iparam]
    handles = []
    idx_N =  np.all((
                    rangeresp>params['minrangeresp'],
                    np.any(corrsig_cells==1,axis=0), #positively modulated cells
                     ),axis=0)
    
    for ialp,alp in enumerate(arealabelpairs):
        ax.errorbar(ialp,np.nanmean(params_regress[idx_N,ialp,iparam]) + 0.1*(ialp<2)*(iparam==0),
                    yerr=stats.sem(params_regress[idx_N,ialp,iparam],nan_policy='omit'),
                    color=clrs_arealabelpairs[ialp],
                    marker='o',markersize=5,linewidth=1.5,elinewidth=1.5,capsize=3)
    ax.legend(handles,legendlabels[2:],frameon=False)
    ax.set_title(['Multiplicative Slope','Additive Offset'][iparam])

    #Test for effect in AL:
    h,p = stats.ttest_ind(params_regress[idx_N,2,iparam],
                            params_regress[idx_N,3,iparam],nan_policy='omit')
    p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
    ax.text(0.75, 0.5, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes)
    ax.set_ylim(ylims[iparam])
    ax_nticks(ax,4)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
for ax in axes:
    ax.set_xticks(range(narealabelpairs),legendlabels,rotation=45,ha='right')
    # ax.set_xlabel('Area-label pair')

my_savefig(fig,savedir,'AL_BarCoefs_StillTrials_%dGRsessions' % (nSessions))

#%% Perform subsampling of the same amount of neurons to balance sample size
# test for an interaction effect: target area determines the effect on slope or additive offset

nresamples = 1000

for iparam in range(2):
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
    table = anova_lm(lm, typ=3) # Type 3 ANOVA (testing interaction first on balanced data)
    for name in ['source','target','source:target']:
        print('%s effect: F=%2.1f,p=%1.3f' % (name,
                                                    table.loc[name,'F'],
                                                    table.loc[name,'PR(>F)']))
    print(table)
    # print('Interaction effect: F=%1.4f,p=%1.4f' % (table.loc['source:target','F'],
    #                                                       table.loc['source:target','PR(>F)']))
    for ialp,alp in enumerate(arealabelpairs):
        print(alp,np.sum(df['arealabelpair'] == alp))
    
    #The number of recorded neurons in AL is smaller than those in V1 and PM:
    #pick nALneurons random entries from the entries that have 'target' set to 'V1PM' to match the number of target categories
    idx_V1PM = np.where(df['target'] == 'V1PM')[0] 
    idx_AL = np.where(df['target'] == 'AL')[0]
    fracsig = np.zeros(nresamples)
    for isub in range(nresamples):
        idx_V1PM_to_pick = np.random.choice(idx_V1PM, size=len(idx_AL), replace=False)

        df_sub = df.iloc[np.concatenate([idx_AL, idx_V1PM_to_pick]),:]

        lm = ols(formula, df_sub).fit()
        table = anova_lm(lm, typ=3) # Type 3 ANOVA DataFrame
        fracsig[isub] = table.loc['source:target','PR(>F)']<=0.05
    print('%2.1f%% of resamples had a significant interaction' % (np.sum(fracsig)/nresamples*100))

# #%%
# ndiffs = 2
# fracmat     = np.full((3,3,narealabelpairs+ndiffs),np.nan)
# nsigmat     = np.full((3,3,narealabelpairs),np.nan)
# ntotalmat   = np.full((3,3,narealabelpairs),np.nan)
# testmat     = np.full((3,3),np.nan)
# ncomparisons = 9
# for ialp,alp in enumerate(arealabelpairs):
#     # for imult, mult in enumerate([-1,0,1]):
#     for imult, mult in enumerate([1,0,-1]):
#         for iadd, add in enumerate([-1,0,1]):
#             idx_N =  np.all((
#                     rangeresp>params['minrangeresp'],
#                     # rangeresp>0.05,
#                     # celldata['nearby'],
#                     # celldata['gOSI']>0.5,
#                     # celldata['noise_level']<maxnoiselevel,
#                      ),axis=0)
#             Nsig = np.sum(np.all((
#                                 sig_params_regress[idx_N,ialp,0]==mult,
#                                 sig_params_regress[idx_N,ialp,1]==add,
#                                 ),axis=0))
#             Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
#             frac = (Nsig/Ntotal) * 100
#             nsigmat[imult,iadd,ialp] = Nsig
#             ntotalmat[imult,iadd,ialp] = Ntotal
#             fracmat[imult,iadd,ialp] = frac
# fracmat[:,:,4] = fracmat[:,:,0] - fracmat[:,:,2]
# fracmat[:,:,5] = fracmat[:,:,1] - fracmat[:,:,3]

# for imult, mult in enumerate([1,0,-1]):
#     for iadd, add in enumerate([-1,0,1]):
#         data = np.array([[nsigmat[imult,iadd,0], ntotalmat[imult,iadd,0]-nsigmat[imult,iadd,0]],
#                          [nsigmat[imult,iadd,1], ntotalmat[imult,iadd,1]-nsigmat[imult,iadd,1]]])
#         testmat[imult,iadd] = stats.chi2_contingency(data)[1]  # p-value
# testmat = testmat * ncomparisons  #bonferroni correction

# fig,axes = plt.subplots(1,narealabelpairs+ndiffs,figsize=((narealabelpairs+ndiffs)*3,3))
# for ialp in range(narealabelpairs+ndiffs):
#     ax = axes[ialp]
#     if ialp < narealabelpairs:
#         vmin,vmax = 0,25
#         # cmap = 'Purples'
#         cmap = 'viridis'
#         # cmap = 'magma'
#         # cmap = 'Greens'
#     else:
#         vmin,vmax = -5,5
#         # cmap = 'bwr'
#         cmap = 'PiYG'
#     im = ax.imshow(fracmat[:,:,ialp],vmin=vmin,vmax=vmax,cmap=cmap)

#     ax.set_xticks([0,1,2],['Sub','None','Add'])
#     # ax.set_yticks([0,1,2],['Div','None','Mult'])
#     ax.set_yticks([0,1,2],['Mult','None','Div'])
#     ax.set_xlabel('Addition')
#     if ialp == 0:
#         ax.set_ylabel('Multiplicative')
#     if ialp < narealabelpairs:
#         ax.set_title(legendlabels[ialp])
#     elif ialp == narealabelpairs:
#         ax.set_title('Diff (FF: PM-AL)')
#     else:
#         ax.set_title('Diff (FB: V1-AL)')
#     # ax.set_title(legendlabels[ialp] if ialp < narealabelpairs else 'Diff (FB-FF)')
#     # ax.set_title(legendlabels[ialp] if ialp < narealabelpairs else 'Diff (FB-FF)')
#     for i in range(3):
#         for j in range(3):
#             if ialp != narealabelpairs:
#                 # ax.text(j,i,'%1.2f' % fracmat[i,j,ialp],ha='center',va='center',color='white' if fracmat[i,j,ialp]<20 else 'black')
#                 ax.text(j,i,'%2.1f%%' % fracmat[i,j,ialp],ha='center',va='center',color='white' if fracmat[i,j,ialp]<20 else 'black')
#             else: 
#                 # ax.text(j,i,'%s%2.1f%%\n%s' % ('+' if fracmat[i,j,ialp]>0 else '',fracmat[i,j,ialp],get_sig_asterisks(testmat[i,j])),ha='center',va='center',color='white' if fracmat[i,j,ialp]>0.2 else 'black')
#                 ax.text(j,i,'%s%2.1f%%\n%s' % ('+' if fracmat[i,j,ialp]>0 else '',fracmat[i,j,ialp],get_sig_asterisks(testmat[i,j])),
#                         # ha='center',va='center',color='white' if fracmat[i,j,ialp]>0.2 else 'black')
#                         ha='center',va='center',color='black')
#                 # ax.text(j,i,'%+2.1f%%' % ('+' if fracmat[i,j,ialp]>0 else '') + '%2.1f%%' % fracmat[i,j,ialp],ha='center',va='center',color='white' if fracmat[i,j,ialp]>0.2 else 'black')

#     fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04,label='% sign. cells')
# plt.tight_layout()
# # sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'Affine_sig_mod_FF_FB_heatmap_%dsessions' % (nSessions))
