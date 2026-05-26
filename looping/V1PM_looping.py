#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress,ranksums
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.session_info import *
from loaddata.get_data_folder import get_local_drive
from utils.gain_lib import * 
from utils.pair_lib import *
from utils.plot_lib import * #get all the fixed color schemes

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\Looping\\')

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
sessions,nSessions   = filter_sessions(protocols = 'GR')
# sessions,nSessions   = filter_sessions(protocols = 'GR',filter_noiselevel=True)
sessiondata          = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=params['calciumversion'])
    # sessions[ises].respmat  /= sessions[ises].celldata['meanF'].to_numpy()[:,None] #convert to deconv/F0

#%% Compute the fraction of nonlooped neurons within a radius from looped cells:
arealabelpairs      = ['PMunlL2/3','V1unlL2/3']
narealabelpairs     = len(arealabelpairs)
legendlabels        = ['PM','V1']

clrs_arealabelpairs = get_clr_arealayers(arealabelpairs) 
# clrs_arealabelpairs = get_clr_arealayers(arealabelpairs[0::2]) 

radii           = np.arange(0,400,20)
# radii           = np.array([0,10,20,50,100,200])
fracnonlooped   = np.zeros((narealabelpairs,len(radii),nSessions))

for ises in range(nSessions):
    for iradius,radius in enumerate(radii):
        sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=radius)
        for ialp,alp in enumerate(arealabelpairs):
            idx_N = sessions[ises].celldata['arealayerlabel'] == alp
            fracnonlooped[ialp,iradius,ises] = np.sum(sessions[ises].celldata['nearby'][idx_N]) / np.sum(idx_N)

#%% Show the fraction of nonlooped neurons within a radius from looped cells:
fig,ax = plt.subplots(1,1,figsize=(5*cm,5*cm),sharex=True,sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    shaded_error(radii,fracnonlooped[ialp,:,:].T,center='mean',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)
    ax.set_ylim([0,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xticks(np.arange(0,400,100))
    ax.set_xlabel('Min dist to looped cell (um)')
    if ialp == 0: 
        ax.set_ylabel('Fraction nonlooped cells')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Fraction_nonlooped_distance_V1PM_%dsessions' % nSessions)

#%% Compute the fraction of looped neurons with at least one neighbour within a radius:
arealabelpairs      = ['PMlabL2/3','V1labL2/3']
narealabelpairs     = len(arealabelpairs)
legendlabels        = ['PM','V1']

radii               = np.arange(0,75,3)
fraclooped          = np.zeros((narealabelpairs,len(radii),nSessions))

for ises in range(nSessions):
    for iradius,radius in enumerate(radii):
        for ialp,alp in enumerate(arealabelpairs):
            idx_N = sessions[ises].celldata['arealayerlabel'] == alp
            idx_L = sessions[ises].celldata['redcell']==0
            idx_nearby = np.any(sessions[ises].distmat_xyz[np.ix_(idx_N,idx_L)]<radius,axis=1)
            fraclooped[ialp,iradius,ises] = np.sum(idx_nearby) / np.sum(idx_N)

#%% Show the fraction of looped neurons that have at least  a radius from looped cells:
fig,ax = plt.subplots(1,1,figsize=(5*cm,5*cm),sharex=True,sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    shaded_error(radii,fraclooped[ialp,:,:].T,center='mean',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)
    ax.set_ylim([0,1])
    ax.set_yticks([0,0.5,1])
    ax.set_xticks(np.arange(0,radii[-1]+10,25))
    ax.set_xlabel('Min dist to looped cell (um)')
    if ialp == 0: 
        ax.set_ylabel('Fraction looped cells included')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Fraction_looped_distance_V1PM_%dsessions' % nSessions)

#%%
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'],metric='xyz')

#%% Show tuning curve when activity in the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'V1lab-V1unl-PMlabL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    'PMlab-PMunl-V1labL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())

mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells),np.nan)

#Regression output:
nregressboots           = 0
# nregressboots           = 250
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

#Correlation output:
# corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
# corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)
# pvals = np.full((narealabelpairs,nSessions),np.nan)

ndprimeboots            = 250
# ndprimeboots = 0
dprimedata              = np.full((narealabelpairs,nCells),np.nan)
dprimesig               = np.full((narealabelpairs,nCells),np.nan)

np.random.seed(0)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])
    
    for ialp,alp in enumerate(arealabelpairs):
        
        # idx_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
        #                                             sessions[ises].celldata['nearby']
        #                                             ),axis=0))[0]
        # idx_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
        #                                             sessions[ises].celldata['nearby']
        #                                             ),axis=0))[0]

        idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_N1) < params['minnneurons'] or len(idx_N2) < params['minnneurons'] or len(idx_N3) < params['minnneurons']:
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
            regressdata[n,:] = linregress(xdata,ydata)[:3]
        params_regress[idx_ses,ialp,:] = regressdata[idx_N3]

        if nregressboots:
            bootregressdata  = np.full((N,nregressboots,3),np.nan)
            bootregress_sig  = np.full((N,2),0)
            for iboot in range(nregressboots):
                meanrespboot            = np.empty([N,len(oris),2])
                for i,ori in enumerate(oris):
                    idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
                    idx_K1              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*params['splitperc']//100,replace=False)
                    idx_K2              = np.random.choice(np.where(idx_T)[0],size=np.sum(idx_T)*params['splitperc']//100,replace=False)
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

        meanresp_pref          = meanresp.copy()
        for n in range(N):
            meanresp_pref[n,:,0] = np.roll(meanresp[n,:,0],-prefori[n])
            meanresp_pref[n,:,1] = np.roll(meanresp[n,:,1],-prefori[n])

        # normalize by peak response
        tempmin,tempmax = meanresp_pref[:,:,0].min(axis=1,keepdims=True),meanresp_pref[:,:,0].max(axis=1,keepdims=True)
        meanresp_pref[:,:,0] = (meanresp_pref[:,:,0] - tempmin) / (tempmax - tempmin)
        meanresp_pref[:,:,1] = (meanresp_pref[:,:,1] - tempmin) / (tempmax - tempmin)

        mean_resp_split_aligned[ialp,:,:,idx_ses] = meanresp_pref[idx_N3]

        # tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        # tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])

        # corrdata_cells[ialp,idx_ses] = tempcorr
        # tempsig = (tempsig<params['alpha_crossrate']) * np.sign(tempcorr)
        # corrsig_cells[ialp,idx_ses] = tempsig

        #dprime metric:
        idx_K1 = np.array([],dtype=int)
        idx_K2 = np.array([],dtype=int)
        for i,ori in enumerate(oris):
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
            idx_K1_T            = np.where(np.logical_and(idx_T,meanpopact < np.nanpercentile(meanpopact[idx_T],params['splitperc'])))[0]
            idx_K1              = np.concatenate((idx_K1,idx_K1_T))
            idx_K2_T            = np.where(np.logical_and(idx_T,meanpopact > np.nanpercentile(meanpopact[idx_T],100-params['splitperc'])))[0]
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

#%%




#%% Fraction of significant multiplicative and additively modulated cells for labeled and unlabeled cells:
modsign = -1
signlabel = 'excited' if modsign==1 else 'inhibited'
orderversion = 1
affinelabels = ['mult.','div.','add.','sub.']

# if orderversion==1:
#     affinelabels = ['mult.','div.','add.','sub.']
# else:
#     affinelabels = ['mult.','add.','div.','sub.']

#%% Make the figure:
clrs_arealabelpairs = ['grey','red','grey','red'] #for unlabeled and labeled cells, respectively
titlelabels = ['FF','FB']
ncomparisons = 4
legendlabels = np.array(['PM$_{ND}$','PM$_{V1}$','V1$_{ND}$','V1$_{PM}$'])

fig,axes = plt.subplots(1,2,figsize=(9*cm,3.8*cm))
idxs = np.array([[0,1],[2,3]]) # FF 0 = unlabeled, 1 = labeled,FB: 2 unlabeled, 3 = labeled
for idir,idx in enumerate(idxs):
    ax = axes[idir]
    sigmat = np.empty((3,2))
    countmat = np.empty((3,2))
    barwidth = 0.5
    for iafftype in [0,1]: # 0 = multiplicative, 1 = additive
        for iaffsign,affsign in enumerate([1,-1]): # 1 = positive, -1 = negative
            fracs = np.empty((2))
            # print(iafftype,affsign)
            for iidx,ialp in enumerate(idx):
                Nsig = np.sum(np.all((
                            dprimesig[ialp,:]==modsign,
                            sig_params_regress[:,ialp,iafftype]==affsign,
                            rangeresp>params['minrangeresp'],
                            celldata['nearby'],
                                ),axis=0))
                Ntotal = np.sum(np.all((
                            ~np.isnan(sig_params_regress[:,ialp,iafftype]),
                            dprimesig[ialp,:]==modsign,
                            rangeresp>params['minrangeresp'],
                            celldata['nearby'],
                                ),axis=0))
                sigmat[iafftype,iidx] = Nsig
                countmat[iafftype,iidx] = Ntotal
                fracs[iidx] = Nsig/Ntotal

                if orderversion==1: 
                    xpos = iaffsign*2 + iafftype*4 + (iidx*2-1) * barwidth/1.5
                else: 
                    xpos = iafftype*2 + iaffsign*4 + (iidx*2-1) * barwidth/1.5
                # print(xpos)
                ax.bar(xpos,fracs[iidx],width=barwidth,color=clrs_arealabelpairs[ialp],edgecolor='k',linewidth=0.5)
            
            if np.any(fracs>0):
                pval = stats.chi2_contingency([[sigmat[iafftype,0], countmat[iafftype,0]-sigmat[iafftype,0]],
                                    [sigmat[iafftype,1], countmat[iafftype,1]-sigmat[iafftype,1]]])[1]
                pval = np.clip(pval*ncomparisons,0,1)
                if orderversion==1: 
                    xpos = iaffsign*2 + iafftype*4 + (np.array([0,1])*2-1) * barwidth/1.5
                else: 
                    xpos = iafftype*2 + iaffsign*4 + (np.array([0,1])*2-1) * barwidth/1.5
                add_stat_annotation(ax,xpos[0],xpos[1],np.max(fracs)+0.05,pval,h=0,fontsize=7)
    ax.legend([plt.Rectangle((0,0),1,1,color=clrs_arealabelpairs[i]) for i in range(2)],
              ['%s (n=%d)' % (legendlabels[i],countmat[0,i]) for i in range(2)],loc='best',fontsize=5) #legendlabels[idx] (),title='n=%d cells' % np.sum(countmat))
    ax_nticks(ax,4)
    ax.set_xticks(np.arange(4)*2,affinelabels)
    ax.set_ylabel('Fraction of %s neurons' % signlabel)
    ax.set_title(titlelabels[idir])
    ax.set_yticks(np.arange(0,1.1,0.2))
    ax.set_ylim([0,1])  
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=1,trim=False)
# my_savefig(fig,savedir,'Looped_FF_FB_Frac_affinemodulation_%s_cells_%dsessions' % (signlabel,nSessions))

#%% Cumulative coefficients
idxs = np.array([[0,1],[2,3]]) # FF 0 = unlabeled, 1 = labeled,FB: 2 unlabeled, 3 = labeled

titlelabels = ['FF','FB']
legendlabels = ['PM$_{ND}$','PM$_{V1}$','V1$_{ND}$','V1$_{PM}$']
ncomparisons = 2
for modsign in [-1,1]:
    signlabel = 'excited' if modsign==1 else 'inhibited'

    fig,axes = plt.subplots(2,2,figsize=(8*cm,8*cm),sharey=True)
    for idir,idx in enumerate(idxs):
        for iparam in range(2):
            ax = axes[idir,iparam]
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
            for ialp in idx:
                idx_N = np.all((
                        rangeresp>params['minrangeresp'],
                        celldata['nearby'],
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
            ax.set_title('%s-%s' % (titlelabels[idir],signlabel))
            h,pval = stats.mannwhitneyu(params_regress[idx_N,idx[0],iparam],
                                    params_regress[idx_N,idx[1],iparam],nan_policy='omit')
            pval = np.clip(pval*ncomparisons,0,1)  #bonferroni + clip

            ax.text(0.45, 0.5, '%s' % (get_sig_asterisks(pval,return_ns=True)),
                    transform=ax.transAxes,fontsize=10)
            ax.set_yticks([0,0.5,1.0])
            ax.set_xlim(xlims)
            ax.set_ylim([0,1])
            ax.set_ylabel('Cumulative fraction of cells')

    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True,offset=2)

    # my_savefig(fig,savedir,'looped_FF_FB_affinemodulation_dprime_%s_cumhistcoefs_%dGRsessions' % (signlabel,nSessions))

#%% 






#%% 
######  ####### ####### #######    ######     #    ######  ### #     #  #####  
#     # #     # #     #    #       #     #   # #   #     #  #  #     # #     # 
#     # #     # #     #    #       #     #  #   #  #     #  #  #     # #       
######  #     # #     #    #       ######  #     # #     #  #  #     #  #####  
#     # #     # #     #    #       #   #   ####### #     #  #  #     #       # 
#     # #     # #     #    #       #    #  #     # #     #  #  #     # #     # 
######  ####### #######    #       #     # #     # ######  ###  #####   #####  

#     # ####### ######  #     # #          #    ####### ####### ######  
##   ## #     # #     # #     # #         # #      #    #       #     # 
# # # # #     # #     # #     # #        #   #     #    #       #     # 
#  #  # #     # #     # #     # #       #     #    #    #####   #     # 
#     # #     # #     # #     # #       #######    #    #       #     # 
#     # #     # #     # #     # #       #     #    #    #       #     # 
#     # ####### ######   #####  ####### #     #    #    ####### ######  

#%% Bootstrapped comparison of correlations and significant correlations with other area: 
# The distribution of correlations is compared to the loop correlation distribution.
# The fraction of significantly positive and negative as well. 
radii           = np.arange(40,100+10,10)
nradii          = len(radii)

idx_PMlab       = np.where(celldata['arealayerlabel'] == 'PMlabL2/3')[0]
idx_V1lab       = np.where(celldata['arealayerlabel'] == 'V1labL2/3')[0]

idx_PMlab       = np.where(np.all((
                                    celldata['arealayerlabel'] == 'PMlabL2/3',
                                    rangeresp>=params['minrangeresp'],
                                    # celldata['noise_level']<params['maxnoiselevel'],
                                    ),axis=0))[0]
idx_V1lab       = np.where(np.all((
                                    celldata['arealayerlabel'] == 'V1labL2/3',
                                    rangeresp>=params['minrangeresp'],
                                    # celldata['noise_level']<params['maxnoiselevel'],
                                    ),axis=0))[0]

nPMlab          = len(idx_PMlab)
nV1lab          = len(idx_V1lab)
idx_PMlab_allnear  = np.full((nPMlab,2,1000),np.nan) #store all unlabeled cell indices within radius
idx_V1lab_allnear  = np.full((nV1lab,2,1000),np.nan)
distancemetric = 'xyz' #or 'xy'

for iN,N in tqdm(enumerate(idx_PMlab),total=nPMlab,desc='Finding PM neurons nearby labeled cells'):
    #get index of which session this labeled cell comes from:
    ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
    if distancemetric == 'xyz': 
        distmat = sessions[ises].distmat_xyz
    elif distancemetric == 'xy':
        distmat = sessions[ises].distmat_xy
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    #get index of labeled cell in this session
    idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
    #get index of all unlabeled cells in this session that are nearby this particular labeled cell
    idx_nearby_ses = np.where(np.all((
                                        np.squeeze(distmat[idx_N_ses,:]<np.max(radii)),
                                        rangeresp[idx_ses]>=params['minrangeresp'],
                                        # sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        # sessions[ises].celldata['arealayerlabel'] == 'PMunlL2/3',
                                        ),axis=0))[0]
    
    # idx_nearby_ses = idx_nearby_ses[:200] #take first 200 closest nearby cells (is all of them actually)
    idx_nearby_all = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
    idx_PMlab_allnear[iN,0,:len(idx_nearby_ses)] = idx_nearby_all
    idx_PMlab_allnear[iN,1,:len(idx_nearby_ses)] = distmat[np.ix_(idx_N_ses,idx_nearby_ses)]

for iN,N in tqdm(enumerate(idx_V1lab),total=nV1lab,desc='Finding V1 neurons nearby labeled cells'):
    #get index of which session this labeled cell comes from:
    ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
    if distancemetric == 'xyz': 
        distmat = sessions[ises].distmat_xyz
    elif distancemetric == 'xy':
        distmat = sessions[ises].distmat_xy
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    #get index of labeled cell in this session
    idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
    #get index of all unlabeled cells in this session that are nearby this particular labeled cell
    idx_nearby_ses = np.where(np.all((
                                        np.squeeze(distmat[idx_N_ses,:]<np.max(radii)),
                                        rangeresp[idx_ses]>=params['minrangeresp'],
                                        # sessions[ises].celldata['arealayerlabel'] == 'V1unlL2/3',
                                        # sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        ),axis=0))[0]
    # idx_nearby_ses = idx_nearby_ses[:200] #limit to first 100 nearby cells
    idx_nearby_all = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
    idx_V1lab_allnear[iN,0,:len(idx_nearby_ses)] = idx_nearby_all
    idx_V1lab_allnear[iN,1,:len(idx_nearby_ses)] = distmat[np.ix_(idx_N_ses,idx_nearby_ses)]


# %% 
# nboots          = 250
nboots          = 1000
np.random.seed(1)

loopfrac        = np.full((2,3,nradii),np.nan) # FF vs FB, +corr vs -corr vs modulated
loopmean        = np.full((2,nradii),np.nan) # FF vs FB, +corr vs -corr
loopmean_abs    = np.full((2,nradii),np.nan) # FF vs FB, +corr vs -corr

bootfrac        = np.full((2,3,nradii,nboots),np.nan) # FF vs FB, +corr vs -corr
bootmean        = np.full((2,nradii,nboots),np.nan) # FF vs FB, +corr vs -corr
bootmean_abs    = np.full((2,nradii,nboots),np.nan) # FF vs FB, +corr vs -corr

binedges        = np.linspace(-1,1,25)
nhistbins       = len(binedges)-1
loophist        = np.full((2,nhistbins,nradii),np.nan) # FF vs FB, +corr vs -corr
boothist        = np.full((2,nhistbins,nradii,nboots),np.nan) # FF vs FB, +corr vs -corr

# loopaffine      = np.full((2,2,nradii),np.nan) # FF vs FB, additive vs multiplicative
# bootaffine      = np.full((2,2,nradii,nboots),np.nan) # FF vs FB, additive vs multiplicative

nhasnearby      = np.full((2,nradii),np.nan) #number of labeled cells that have at least min_nearby unlabeled cells within radius
min_nearby      = 2

sigdata = dprimesig
moddata = dprimedata

for irad,radius in enumerate(tqdm(radii,total=nradii,desc='Bootstrapping for different radii')):
    #For PMlab:
    hasnearby = np.full(len(idx_PMlab),False)
    iapl = 0
    for iN,N in enumerate(idx_PMlab):
        idx_within_radius = idx_PMlab_allnear[iN,0,idx_PMlab_allnear[iN,1,:] <= radius]
        if len(idx_within_radius) >= min_nearby:
            hasnearby[iN] = True
    idx_PMlab_haswithinradius   = idx_PMlab[hasnearby].astype(int)
    # print('PMlab: %d/%d' % (np.sum(hasnearby),len(hasnearby)))
    nhasnearby[iapl,irad] = np.sum(hasnearby)

    loopfrac[iapl,0,irad] = np.sum(sigdata[1,idx_PMlab_haswithinradius]==1) / len(idx_PMlab_haswithinradius)
    loopfrac[iapl,1,irad] = np.sum(sigdata[1,idx_PMlab_haswithinradius]==-1) / len(idx_PMlab_haswithinradius)
    loopfrac[iapl,2,irad] = (loopfrac[iapl,0,irad]+loopfrac[iapl,1,irad])
    
    loopmean[iapl,irad] = np.nanmean(moddata[1,idx_PMlab_haswithinradius])
    loopmean_abs[iapl,irad] = np.nanmean(np.abs(moddata[1,idx_PMlab_haswithinradius]))

    histcounts              = np.histogram(moddata[1,idx_PMlab_haswithinradius],bins=binedges)[0]
    loophist[iapl,:,irad]   = np.cumsum(histcounts)/np.sum(histcounts)

    # loopaffine[iapl,0,irad] = np.nanmean(params_regress[idx_PMlab_haswithinradius,ialp,0]) #multiplicative slope
    # loopaffine[iapl,1,irad] = np.nanmean(params_regress[idx_PMlab_haswithinradius,ialp,1]) #additive offset

    for iboot in range(nboots):
        idx_boot = np.full(len(idx_PMlab),np.nan)
        for iN,N in enumerate(idx_PMlab):
            idx_within_radius = idx_PMlab_allnear[iN,0,idx_PMlab_allnear[iN,1,:] <= radius]
            if hasnearby[iN]:
                idx_boot[iN] = np.random.choice(idx_within_radius,1)
        idx_boot                    = idx_boot[hasnearby].astype(int)
        bootfrac[iapl,0,irad,iboot] = np.sum(sigdata[0,idx_boot]==1) / len(idx_boot) #compute fraction of sig pos for this boot
        bootfrac[iapl,1,irad,iboot] = np.sum(sigdata[0,idx_boot]==-1) / len(idx_boot)
        bootfrac[iapl,2,irad,iboot] = (bootfrac[iapl,0,irad,iboot]+bootfrac[iapl,1,irad,iboot])

        bootmean[iapl,irad,iboot] = np.nanmean(moddata[0,idx_boot])
        bootmean_abs[iapl,irad,iboot] = np.nanmean(np.abs(moddata[0,idx_boot]))

        histcounts = np.histogram(moddata[0,idx_boot],bins=binedges)[0]
        boothist[iapl,:,irad,iboot] = np.cumsum(histcounts)/np.sum(histcounts)

        # bootaffine[iapl,0,irad,iboot] = np.nanmean(params_regress[idx_boot,ialp,0]) #multiplicative slope
        # bootaffine[iapl,1,irad,iboot] = np.nanmean(params_regress[idx_boot,ialp,1]) #additive offset

    #Now for V1lab:
    iapl = 1
    hasnearby = np.full(len(idx_V1lab),False)

    for iN,N in enumerate(idx_V1lab):
        idx_within_radius = idx_V1lab_allnear[iN,0,idx_V1lab_allnear[iN,1,:] <= radius]
        if len(idx_within_radius) >= min_nearby:
            hasnearby[iN] = True
    idx_V1lab_haswithinradius   = idx_V1lab[hasnearby].astype(int)
    # print('V1lab: %d/%d' % (np.sum(hasnearby),len(hasnearby)))
    nhasnearby[iapl,irad] = np.sum(hasnearby)

    loopfrac[iapl,0,irad] = np.sum(sigdata[3,idx_V1lab_haswithinradius]==1) / len(idx_V1lab_haswithinradius)
    loopfrac[iapl,1,irad] = np.sum(sigdata[3,idx_V1lab_haswithinradius]==-1) / len(idx_V1lab_haswithinradius)
    loopfrac[iapl,2,irad] = (loopfrac[iapl,0,irad]+loopfrac[iapl,1,irad])
    
    loopmean[iapl,irad] = np.nanmean(moddata[3,idx_V1lab_haswithinradius])
    loopmean_abs[iapl,irad] = np.nanmean(np.abs(moddata[3,idx_V1lab_haswithinradius]))
    histcounts      = np.histogram(moddata[3,idx_V1lab_haswithinradius],bins=binedges)[0]
    loophist[iapl,:,irad]   = np.cumsum(histcounts)/np.sum(histcounts)

    # loopaffine[iapl,0,irad] = np.nanmean(params_regress[idx_V1lab_haswithinradius,ialp,0]) #multiplicative slope
    # loopaffine[iapl,1,irad] = np.nanmean(params_regress[idx_V1lab_haswithinradius,ialp,1]) #additive offset

    for iboot in range(nboots):
        idx_boot = np.full(len(idx_V1lab),np.nan)
        for iN,N in enumerate(idx_V1lab):
            idx_within_radius = idx_V1lab_allnear[iN,0,idx_V1lab_allnear[iN,1,:] <= radius]
            if hasnearby[iN]:
                idx_boot[iN] = np.random.choice(idx_within_radius,1)
        idx_boot                    = idx_boot[hasnearby].astype(int)
        bootfrac[iapl,0,irad,iboot] = np.sum(sigdata[2,idx_boot]==1) / len(idx_boot) #compute fraction of sig pos for this boot
        bootfrac[iapl,1,irad,iboot] = np.sum(sigdata[2,idx_boot]==-1) / len(idx_boot)
        bootfrac[iapl,2,irad,iboot] = (bootfrac[iapl,0,irad,iboot]+bootfrac[iapl,1,irad,iboot])

        bootmean[iapl,irad,iboot] = np.nanmean(moddata[2,idx_boot])
        bootmean_abs[iapl,irad,iboot] = np.nanmean(np.abs(moddata[2,idx_boot]))
        histcounts = np.histogram(moddata[2,idx_boot],bins=binedges)[0]
        boothist[iapl,:,irad,iboot] = np.cumsum(histcounts)/np.sum(histcounts)

        # bootaffine[iapl,0,irad,iboot] = np.nanmean(params_regress[idx_boot,ialp,0]) #multiplicative slope
        # bootaffine[iapl,1,irad,iboot] = np.nanmean(params_regress[idx_boot,ialp,1]) #additive offset

#%% Plotting bootstrapped results across different radii:

legendlabels        = ['FF','FB']
axisbuffer          = 0.005
lw                  = 2

subplotlabels       = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.','Frac. Mod.'])
loopdata_subplots   = np.stack((loopmean,loopmean_abs,loopfrac[:,0],loopfrac[:,1],loopfrac[:,2]),axis=2)
bootdata_subplots   = np.stack((bootmean,bootmean_abs,bootfrac[:,0],bootfrac[:,1],bootfrac[:,2]),axis=2)
nmetrics            = len(subplotlabels)
params['ci']        = 95
fig,axes = plt.subplots(2,nmetrics,figsize=(nmetrics*3.3*cm,6*cm))
clrs_arealabelpairs = [ '#9933FF','#00CC99']

for ialp in range(2):
    for imetric in range(nmetrics):
        axidx = imetric
        ax  = axes[ialp,axidx]
        # ax.plot(radii,loopdata_subplots[ialp,:,imetric],color=clrs_arealabelpairs[ialp],linewidth=lw,marker='.',markersize=8)
        ax.plot(radii,loopdata_subplots[ialp,:,imetric],color='r',linewidth=lw,marker='.',markersize=8)
        tempdata = bootdata_subplots[np.ix_([ialp],range(nradii),[imetric],range(nboots))].squeeze()
        
        ax.fill_between(radii, np.percentile(tempdata,(100-params['ci'])/2,axis=1), np.percentile(tempdata,params['ci']+(100-params['ci'])/2,axis=1), color='grey',alpha=0.25)
        ax_nticks(ax,3)
        # for irad,radius in enumerate(radii):
            # ax.text((irad+1)/len(radii),0,'n=%d'%nhasnearby[ialp,irad],fontsize=6,
            #         ha='center',va='bottom',transform=ax.transAxes,color='grey',rotation=45)
        for irad,radius in enumerate(radii):
            pval = np.sum(loopdata_subplots[ialp,irad,imetric] > bootdata_subplots[ialp,irad,imetric,:]) / nboots
            pval = np.min([pval,1-pval])
            # ax.text((irad+1)/len(radii),loopdata_subplots[ialp,irad,imetric]+axisbuffer,get_sig_asterisks(pval),
            ax.text(radius,loopdata_subplots[ialp,irad,imetric]-axisbuffer,get_sig_asterisks(pval),
                    fontsize=7,ha='center',va='top',color='k',rotation=45)
            # print(pval)
        axes[ialp,axidx].set_title(legendlabels[ialp] + ' ' + subplotlabels[imetric])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Looped_Modulations_Bootstrap_Radii_dprime')
# my_savefig(fig,savedir,'Looped_Modulations_Bootstrap_Radii_corr')

#%% Plotting bootstrapped results at a specific radius:
radius              = 50
irad                = np.where(radii==radius)[0][0]

legendlabels        = ['FF','FB']
axisbuffer          = 0.02
lw                  = 2
nbins               = 20
subplotlabels       = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.','Frac. Mod.'])
loopdata_subplots   = np.stack((loopmean[:,irad],loopmean_abs[:,irad],loopfrac[:,0,irad],loopfrac[:,1,irad],loopfrac[:,2,irad]),axis=1)
bootdata_subplots   = np.stack((bootmean[:,irad],bootmean_abs[:,irad],bootfrac[:,0,irad],bootfrac[:,1,irad],bootfrac[:,2,irad]),axis=1)
nmetrics            = len(subplotlabels)

fig,axes = plt.subplots(2,nmetrics+1,figsize=(nmetrics*3.3*cm,6*cm))
ncells = [nPMlab,nV1lab]
for ialp in range(2):
    axes[ialp,0].plot(binedges[:-1],loophist[ialp,:,irad],color=clrs_arealabelpairs[ialp])
    tempdata = boothist[np.ix_([ialp],range(nhistbins),[irad],range(nboots))].squeeze()
    shaded_error(binedges[:-1],np.nanmean(tempdata,axis=1),np.nanstd(tempdata,axis=1),
                    ax=axes[ialp,0],color='grey')
    axes[ialp,0].set_xlim([binedges[np.where(loophist[ialp,:]>0)[0][0]],binedges[np.where(loophist[ialp,:]>0.999)[0][0]]])
    axes[ialp,0].set_ylim([0,1])
    axes[ialp,0].set_ylabel(legendlabels[ialp],fontsize=7,fontweight='bold',color=clrs_arealabelpairs[ialp])
    if ialp == 0:
        # axes[ialp,0].set_title('Corr. coeff.')
        axes[ialp,0].set_title('Dprime')
    
    for imetric in range(nmetrics):
        axidx = imetric+1
        ax = axes[ialp,axidx]
        ax.axvline(loopdata_subplots[ialp,imetric],color='red',linewidth=lw)
        bootmin,bootmax = np.percentile(bootdata_subplots[ialp,imetric],(0,100))
        bootdiff = 100/(ncells[ialp]-2)/100
        # bins = np.linspace(bootmin-axisbuffer,bootmax+axisbuffer,nbins)
        bins = np.arange(bootmin,bootmax,bootdiff)
        sns.histplot(bootdata_subplots[ialp,imetric,:],ax=ax,bins=bins,element='step',stat='probability',color='grey')
        xlims = [np.min([np.percentile(bootdata_subplots[ialp,imetric],0),loopdata_subplots[ialp,imetric]])-axisbuffer,
                 np.max([np.percentile(bootdata_subplots[ialp,imetric],100),loopdata_subplots[ialp,imetric]])+axisbuffer]
        ax.set_xlim(xlims) #set lim to extremes of bootstrapped data + small buffer
        # pval = np.sum(bootdata_subplots[ialp,imetric,:]>loopdata_subplots[ialp,imetric])/len(bootdata_subplots[ialp,imetric,:])
        pval = np.sum(loopdata_subplots[ialp,imetric] > bootdata_subplots[ialp,imetric,:]) / nboots
        pval = np.min([pval,1-pval])
        ax.text(loopdata_subplots[ialp,imetric],ax.get_ylim()[1]*0.8,get_sig_asterisks(pval,return_ns=False),fontsize=11,
                color='red',ha='right',va='center',fontweight='bold')
                # color=clrs_arealabelpairs[ialp])
        if ialp == 0:
            ax.set_title(subplotlabels[imetric])
        ax_nticks(ax,3)
        ax.set_ylabel('')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Looped_Dprime_Modulations_Bootstrap_Radius%d' % (radius))



#%% Evaluating potential confound of anatomical proximity:
# To evaluate potential confounds due to anatomical inhomogeneities in modulations, we first measured the absolute difference in 
# modulation as a function of distance between pairs of cells, regardless of their projection target. We found that cells that are
# in close anatomical proximity are modulated more similarly (Fig. S10a,b).

step                = 20
binedges            = np.arange(0,300 + step,step)
bincenters          = binedges[:-1] + step/2
nbins              = len(bincenters)
deltadprime         = np.full((2,nbins,nSessions),np.nan) #store absolute difference in modulation between pairs of cells for each session
# deltadprime         = np.full((2,nradii,nSessions),np.nan) #store absolute difference in modulation between pairs of cells for each session
moddata = dprimedata

for ises in tqdm(range(nSessions),desc='Computing modulation difference as a function of distance'):
    idx_ses = np.isin(celldata['session_id'],sessions[ises].session_id)
    #get pairs of cells that are within this radius of each other:
    if distancemetric == 'xyz': 
        distmat = sessions[ises].distmat_xyz
    elif distancemetric == 'xy':
        distmat = sessions[ises].distmat_xy
    deltadprime_FF = np.abs(moddata[0,idx_ses][:,None] - moddata[0,idx_ses][None,:])
    deltadprime_FB = np.abs(moddata[2,idx_ses][:,None] - moddata[2,idx_ses][None,:])
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

#statistics: 
df = nd_array_to_dataframe(deltadprime,dim_names=['areapair','distance', 'session'])
formula = "value ~ distance*areapair + C(session)" #include session as random effect to account for repeated measures across sessions
lm = ols(formula, df).fit()
table = anova_lm(lm, typ=2) # Type 2 ANOVA
for name in ['distance','areapair','distance:areapair']:
    print('%s effect: F=%2.1f,p=%1.3f' % (name,table.loc[name,'F'],
                                                table.loc[name,'PR(>F)']))
print(table)
# if table.loc['areapair','PR(>F)'] < 0.05:
ax.text(0.05,0.96,'Pathway:%s' % get_sig_asterisks(table.loc['areapair','PR(>F)']),
            fontsize=5,transform=ax.transAxes,ha='left',va='center')
# if table.loc['distance','PR(>F)'] < 0.05:
ax.text(0.05,0.88,'Distance:%s' % get_sig_asterisks(table.loc['distance','PR(>F)']),
            fontsize=5,transform=ax.transAxes,ha='left',va='center')
# if table.loc['distance:areapair','PR(>F)'] < 0.05:
ax.text(0.05,0.8,'Interaction:%s' % get_sig_asterisks(table.loc['distance:areapair','PR(>F)'],return_ns=True),
            fontsize=5,transform=ax.transAxes,ha='left',va='center')

sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Delta_dprime_distance_V1PM_%dsessions' % nSessions)

#%% Compute the fraction of looped neurons with at least two neighbours within a radius:
legendlabels        = ['PM','V1']
radii               = np.arange(0,110,10)
fraclooped          = np.zeros((2,len(radii)))

for iradius,radius in enumerate(radii):
    fraclooped[0,iradius] = np.sum(np.sum(idx_PMlab_allnear[:,1,:]<=radius,axis=1)>=min_nearby) / np.shape(idx_PMlab)[0]
    fraclooped[1,iradius] = np.sum(np.sum(idx_V1lab_allnear[:,1,:]<=radius,axis=1)>=min_nearby) / np.shape(idx_V1lab)[0]

#%% Show the fraction of nonlooped neurons within a radius from looped cells:
fig,ax = plt.subplots(1,1,figsize=(4*cm,4*cm),sharex=True,sharey=True)
for ialp,alp in enumerate(legendlabels):
    ax.plot(radii,fraclooped[ialp,:],color=clrs_arealabelpairs[ialp],linewidth=1.5)
    ax.set_ylim([0,1])
    ax.set_yticks([0,0.5,1])
    # ax.set_xticks(radii)
    ax.set_xticks(np.arange(0,radii[-1]+10,25))
    ax.set_xlabel('Radius (um)')
    if ialp == 0: 
        ax.set_ylabel('Fraction looped cells included')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Fraction_looped_2respneighbours_distance_V1PM_%dsessions' % nSessions)

#%% Compute the number of nonlooped neurons within radius
legendlabels        = ['PM','V1']
nNonlooped          = np.zeros((2,len(radii)))

for iradius,radius in enumerate(radii):
    nNonlooped[0,iradius] = np.nanmean(np.sum(idx_PMlab_allnear[:,1,:]<=radius,axis=1))
    nNonlooped[1,iradius] = np.nanmean(np.sum(idx_V1lab_allnear[:,1,:]<=radius,axis=1))

#%% Show the fraction of nonlooped neurons within a radius from looped cells:
fig,ax = plt.subplots(1,1,figsize=(4*cm,4*cm),sharex=True,sharey=True)
for ialp,alp in enumerate(legendlabels):
    ax.plot(radii,nNonlooped[ialp,:],color=clrs_arealabelpairs[ialp])
    # ax.set_ylim([0,1])
    ax.set_xticks(np.arange(0,radii[-1]+100,100))
    ax.set_xlabel('Radius (um)')
    if ialp == 0: 
        ax.set_ylabel('# nonlooped cells')
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Number_nonlooped_distance_V1PM_%dsessions' % nSessions)

#%% Show an example plane: 
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
for ises in range(nSessions):
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    # sessions[ises].celldata['modulation'] = np.nanmean(corrdata_cells[:,idx_ses],axis=0)
    sessions[ises].celldata['dprime'] = np.nanmean(dprimedata[:,idx_ses],axis=0)
    sessions[ises].celldata['rangeresp'] = rangeresp[idx_ses]
    sessions[ises].celldata['sigmod'] = np.nanmean(dprimesig[:,idx_ses],axis=0)
    # sessions[ises].celldata['sigmod'] = np.nanmean(dprimesig[:,idx_ses],axis=0)

#%% 
for ises in range(nSessions):
    # print('Session %d: %d modulated PM looped cells' % (ises,sum(np.logical_and(sessions[ises].celldata['sigmod']!=0,sessions[ises].celldata['arealabel']=='PMlab'))))
    print('Session %d: %d modulated PM looped cells' % (ises,sum(np.logical_and(sessions[ises].celldata['sigmod']==1,
                                                                                sessions[ises].celldata['arealayerlabel']=='PMlabL2/3'))))
    # print('Session %d: %d modulated V1 looped cells' % (ises,sum(np.logical_and(sessions[ises].celldata['sigmod']!=0,sessions[ises].celldata['arealabel']=='V1lab'))))

#%%
ises = 7
iplane = 4
fig = plot_loop_plane(sessions[ises].celldata,iplane=iplane,filter_rangeresp=True,id_looped=True,radiuslooped=True,
                     cellfield='dprime',radius=50) 
filename = 'Example_plane_dprime_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,'ExamplePlanes',filename),format = 'pdf',dpi=600,bbox_inches='tight')

#%%
ises = 5
iplane = 1
fig = plot_loop_plane(sessions[ises].celldata,iplane=iplane,filter_rangeresp=True,id_looped=True,radiuslooped=True,
                     cellfield='dprime',radius=50) 
filename = 'Example_plane_dprime_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,'ExamplePlanes',filename),format = 'pdf',dpi=600,bbox_inches='tight')

#%% 
######  ####### ####### #######    ######     #    ######  ### #     #  #####  
#     # #     # #     #    #       #     #   # #   #     #  #  #     # #     # 
#     # #     # #     #    #       #     #  #   #  #     #  #  #     # #       
######  #     # #     #    #       ######  #     # #     #  #  #     #  #####  
#     # #     # #     #    #       #   #   ####### #     #  #  #     #       # 
#     # #     # #     #    #       #    #  #     # #     #  #  #     # #     # 
######  ####### #######    #       #     # #     # ######  ###  #####   #####  
   #    ####### ####### ### #     # #######                                    
  # #   #       #        #  ##    # #                                          
 #   #  #       #        #  # #   # #                                          
#     # #####   #####    #  #  #  # #####                                      
####### #       #        #  #   # # #                                          
#     # #       #        #  #    ## #                                          
#     # #       #       ### #     # #######                                    

#%% Bootstrapped comparison of correlations and significant correlations with other area: 
# The distribution of correlations is compared to the loop correlation distribution.
# The fraction of significantly positive and negative as well. 
radii           = np.arange(40,100+10,10)
nradii          = len(radii)

idx_all = np.all((
                # celldata['noise_level']<params['maxnoiselevel'],
                rangeresp>=params['minrangeresp'],
                np.any(dprimesig==1,axis=0),
                ),axis=0)

idx_PMlab       = np.where(np.all((
                                    celldata['arealayerlabel'] == 'PMlabL2/3',
                                    idx_all
                                    ),axis=0))[0]
idx_V1lab       = np.where(np.all((
                                    celldata['arealayerlabel'] == 'V1labL2/3',
                                    idx_all
                                    ),axis=0))[0]

nPMlab          = len(idx_PMlab)
nV1lab          = len(idx_V1lab)
idx_PMlab_allnear  = np.full((nPMlab,2,1000),np.nan) #store all unlabeled cell indices within radius
idx_V1lab_allnear  = np.full((nV1lab,2,1000),np.nan)
distancemetric  = 'xyz' #or 'xy'

for iN,N in tqdm(enumerate(idx_PMlab),total=nPMlab,desc='Finding PM neurons nearby labeled cells'):
    #get index of which session this labeled cell comes from:
    ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
    if distancemetric == 'xyz': 
        distmat = sessions[ises].distmat_xyz
    elif distancemetric == 'xy':
        distmat = sessions[ises].distmat_xy
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    #get index of labeled cell in this session
    idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
    #get index of all unlabeled cells in this session that are nearby this particular labeled cell
    idx_nearby_ses = np.where(np.all((
                                        np.squeeze(distmat[idx_N_ses,:]<np.max(radii)),
                                        sessions[ises].celldata['arealayerlabel'] == 'PMunlL2/3',
                                        idx_all[idx_ses]
                                        # rangeresp[idx_ses]>=params['minrangeresp'],
                                        # sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        # np.any(dprimesig[:,idx_ses]==1,axis=0)
                                        ),axis=0))[0]
    
    # idx_nearby_ses = idx_nearby_ses[:200] #take first 200 closest nearby cells (is all of them actually)
    idx_nearby_all = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
    idx_PMlab_allnear[iN,0,:len(idx_nearby_ses)] = idx_nearby_all
    idx_PMlab_allnear[iN,1,:len(idx_nearby_ses)] = distmat[np.ix_(idx_N_ses,idx_nearby_ses)]

for iN,N in tqdm(enumerate(idx_V1lab),total=nV1lab,desc='Finding V1 neurons nearby labeled cells'):
    #get index of which session this labeled cell comes from:
    ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
    if distancemetric == 'xyz': 
        distmat = sessions[ises].distmat_xyz
    elif distancemetric == 'xy':
        distmat = sessions[ises].distmat_xy
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    #get index of labeled cell in this session
    idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
    #get index of all unlabeled cells in this session that are nearby this particular labeled cell
    idx_nearby_ses = np.where(np.all((
                                        np.squeeze(distmat[idx_N_ses,:]<np.max(radii)),
                                        sessions[ises].celldata['arealayerlabel'] == 'V1unlL2/3',
                                        idx_all[idx_ses]
                                        # rangeresp[idx_ses]>=params['minrangeresp'],
                                        # sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        # np.any(dprimesig[:,idx_ses]==1,axis=0)
                                        ),axis=0))[0]
    # idx_nearby_ses = idx_nearby_ses[:200] #limit to first 100 nearby cells
    idx_nearby_all = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
    idx_V1lab_allnear[iN,0,:len(idx_nearby_ses)] = idx_nearby_all
    idx_V1lab_allnear[iN,1,:len(idx_nearby_ses)] = distmat[np.ix_(idx_N_ses,idx_nearby_ses)]

#%% 
nboots          = 50
nboots          = 1000
np.random.seed(0)

loopaffine      = np.full((2,2,nradii),np.nan) # FF vs FB, additive vs multiplicative
bootaffine      = np.full((2,2,nradii,nboots),np.nan) # FF vs FB, additive vs multiplicative

nhasnearby      = np.full((2,nradii),np.nan) #number of labeled cells that have at least min_nearby unlabeled cells within radius
min_nearby      = 2

for irad,radius in enumerate(tqdm(radii,total=nradii,desc='Bootstrapping for different radii')):
    #For PMlab:
    hasnearby = np.full(len(idx_PMlab),False)
    iapl = 0
    for iN,N in enumerate(idx_PMlab):
        idx_within_radius = idx_PMlab_allnear[iN,0,idx_PMlab_allnear[iN,1,:] <= radius]
        if len(idx_within_radius) >= min_nearby:
            hasnearby[iN] = True
    idx_PMlab_haswithinradius   = idx_PMlab[hasnearby].astype(int)
    # print('PMlab: %d/%d' % (np.sum(hasnearby),len(hasnearby)))
    nhasnearby[iapl,irad] = np.sum(hasnearby)
    # print('Radius %d: %d PM looped cells with at least %d nearby cells' % (radius,np.sum(hasnearby),min_nearby))
    
    loopaffine[iapl,0,irad] = np.nanmean(params_regress[idx_PMlab_haswithinradius,:,0]) #multiplicative slope
    loopaffine[iapl,1,irad] = np.nanmean(params_regress[idx_PMlab_haswithinradius,:,1]) #additive offset

    for iboot in range(nboots):
        idx_boot = np.full(len(idx_PMlab),np.nan)
        for iN,N in enumerate(idx_PMlab):
            idx_within_radius = idx_PMlab_allnear[iN,0,idx_PMlab_allnear[iN,1,:] <= radius]
            if hasnearby[iN]:
                idx_boot[iN] = np.random.choice(idx_within_radius,1)
        idx_boot                    = idx_boot[hasnearby].astype(int)

        bootaffine[iapl,0,irad,iboot] = np.nanmean(params_regress[idx_boot,:,0]) #multiplicative slope
        bootaffine[iapl,1,irad,iboot] = np.nanmean(params_regress[idx_boot,:,1]) #additive offset

    #Now for V1lab:
    iapl = 1
    hasnearby = np.full(len(idx_V1lab),False)

    for iN,N in enumerate(idx_V1lab):
        idx_within_radius = idx_V1lab_allnear[iN,0,idx_V1lab_allnear[iN,1,:] <= radius]
        if len(idx_within_radius) >= min_nearby:
            hasnearby[iN] = True
    idx_V1lab_haswithinradius   = idx_V1lab[hasnearby].astype(int)
    # print('V1lab: %d/%d' % (np.sum(hasnearby),len(hasnearby)))
    nhasnearby[iapl,irad] = np.sum(hasnearby)

    # print('Radius %d: %d V1 looped cells with at least %d nearby cells' % (radius,np.sum(hasnearby),min_nearby))

    loopaffine[iapl,0,irad] = np.nanmean(params_regress[idx_V1lab_haswithinradius,:,0]) #multiplicative slope
    loopaffine[iapl,1,irad] = np.nanmean(params_regress[idx_V1lab_haswithinradius,:,1]) #additive offset

    for iboot in range(nboots):
        idx_boot = np.full(len(idx_V1lab),np.nan)
        for iN,N in enumerate(idx_V1lab):
            idx_within_radius = idx_V1lab_allnear[iN,0,idx_V1lab_allnear[iN,1,:] <= radius]
            if hasnearby[iN]:
                idx_boot[iN] = np.random.choice(idx_within_radius,1)
        idx_boot                    = idx_boot[hasnearby].astype(int)

        bootaffine[iapl,0,irad,iboot] = np.nanmean(params_regress[idx_boot,:,0]) #multiplicative slope
        bootaffine[iapl,1,irad,iboot] = np.nanmean(params_regress[idx_boot,:,1]) #additive offset

#%% Plotting bootstrapped results across different radii:
legendlabels        = ['FF','FB']
axisbuffer          = 0.005
lw                  = 2

# subplotlabels       = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.','Frac. Mod.'])
subplotlabels       = np.array(['Slope','Offset'])
loopdata_subplots   = np.stack((loopaffine[:,0],loopaffine[:,1]),axis=2)
bootdata_subplots   = np.stack((bootaffine[:,0],bootaffine[:,1]),axis=2)
nmetrics            = len(subplotlabels)
params['ci']        = 95
fig,axes = plt.subplots(2,nmetrics,figsize=(nmetrics*3.3*cm,6*cm),sharey='col')
clrs_arealabelpairs = [ '#9933FF','#00CC99']

for ialp in range(2):
    for imetric in range(nmetrics):
        axidx = imetric
        ax  = axes[ialp,axidx]
        # ax.plot(radii,loopdata_subplots[ialp,:,imetric],color=clrs_arealabelpairs[ialp],linewidth=lw,marker='.',markersize=8)
        ax.plot(radii,loopdata_subplots[ialp,:,imetric],color='r',linewidth=lw,marker='.',markersize=8)
        tempdata = bootdata_subplots[np.ix_([ialp],range(nradii),[imetric],range(nboots))].squeeze()
        
        ax.fill_between(radii, np.percentile(tempdata,(100-params['ci'])/2,axis=1), np.percentile(tempdata,params['ci']+(100-params['ci'])/2,axis=1), color='grey',alpha=0.25)
        ax_nticks(ax,3)
        for irad,radius in enumerate(radii):
            pval = np.sum(loopdata_subplots[ialp,irad,imetric] > bootdata_subplots[ialp,irad,imetric,:]) / nboots
            pval = np.min([pval,1-pval])
            pval *= nradii
             
            # ax.text((irad+1)/len(radii),loopdata_subplots[ialp,irad,imetric]+axisbuffer,get_sig_asterisks(pval),
            ax.text(radius,loopdata_subplots[ialp,irad,imetric]-axisbuffer,get_sig_asterisks(pval),
                    fontsize=7,ha='center',va='top',color='k',rotation=45)
        axes[ialp,axidx].set_title(legendlabels[ialp] + ' ' + subplotlabels[imetric])
        # if imetric == 0:
            # axes[ialp,axidx].axhline(1,color='grey',linewidth=1,linestyle='--')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Looped_Affine_Bootstrap_Radii_dprime')


#%% 
######  ####### ####### #######    ######     #    ######  ### #     #  #####  
#     # #     # #     #    #       #     #   # #   #     #  #  #     # #     # 
#     # #     # #     #    #       #     #  #   #  #     #  #  #     # #       
######  #     # #     #    #       ######  #     # #     #  #  #     #  #####  
#     # #     # #     #    #       #   #   ####### #     #  #  #     #       # 
#     # #     # #     #    #       #    #  #     # #     #  #  #     # #     # 
######  ####### #######    #       #     # #     # ######  ###  #####   #####  

#     # ### #     #    ######  ###  #####  ####### 
##   ##  #  ##    #    #     #  #  #     #    #    
# # # #  #  # #   #    #     #  #  #          #    
#  #  #  #  #  #  #    #     #  #   #####     #    
#     #  #  #   # #    #     #  #        #    #    
#     #  #  #    ##    #     #  #  #     #    #    
#     # ### #     #    ######  ###  #####     #    

#%% Bootstrapped comparison of significantly modulated neurons as function of the distance to a looped cell. 
radii           = np.arange(20,200+10,10)
nradii          = len(radii)


distancemetric = 'xyz' #or 'xy'

idx_rad = np.full((2, N, nradii-1), np.nan)

for ises in tqdm(range(nSessions),desc='Finding PM neurons nearby labeled cells'):
    #get index of which session this labeled cell comes from:
    # ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
    if distancemetric == 'xyz': 
        distmat = sessions[ises].distmat_xyz
    elif distancemetric == 'xy':
        distmat = sessions[ises].distmat_xy

    mindist = np.nanmin(distmat[sessions[ises].celldata['redcell']==1,:],axis=0)
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)    
    # idx_ses     = np.where(idx_ses)[0]
    # sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=radius)
    for irad, (minrad, maxrad) in enumerate(zip(radii[:-1], radii[1:])):
        # print(irad)

        idx_N = np.all((mindist <= maxrad,
                        mindist > minrad,
                        sessions[ises].celldata['arealayerlabel'] == 'PMunlL2/3',
                        ), axis=0)
        # print(np.sum(idx_N))
        
        idx_rad[0, idx_ses, irad] = idx_N

        idx_N = np.all((mindist <= maxrad,
                        mindist > minrad,
                        sessions[ises].celldata['arealayerlabel'] == 'V1unlL2/3',
                        ), axis=0)
        print(np.sum(idx_N))
        idx_rad[1, idx_ses, irad] = idx_N

#%%
def my_shuffle(data,method='random',axis=0):
    data = copy.deepcopy(data)
    if method == 'random':
        if axis == 0:
            for icol in range(data.shape[1]):
                data[:,icol] = np.random.permutation(data[:,icol])
        elif axis == 1:
            for irow in range(data.shape[0]):
                data[irow,:] = np.random.permutation(data[irow,:])
        elif axis is None:
            rng = np.random.default_rng()
            orig_size = data.shape
            data = np.random.permutation(data.ravel()).reshape(orig_size)

    elif method == 'circular':
        if axis == 0:
            for icol in range(data.shape[1]):
                data[:,icol] = np.roll(data[:,icol],shift=np.random.randint(0,data.shape[0]))
        elif axis == 1:
            for irow in range(data.shape[0]):
                data[irow,:] = np.roll(data[irow,:],shift=np.random.randint(0,data.shape[1])) 
    else:
        raise ValueError('method should be "random" or "circular"')
    return data

#%% Bootstratpped fraction of significantly modulated neurons as a function of minimum distance to looped cell:
nboot           = 100
frac_mod        = np.zeros((2,nradii-1))
frac_mod_boot   = np.zeros((2,nradii-1,nboot))

# for irad in tqdm(range(nradii-1),desc='Bootstrapping for different radii'):
for irad in range(nradii-1):

    summod = np.sum(dprimesig[0,np.where(idx_rad[0, :, irad])[0]] !=0)
    frac_mod[0, irad] = summod / np.sum(idx_rad[0, :, irad])
    summod = np.sum(dprimesig[2,np.where(idx_rad[1, :, irad])[0]] !=0)
    frac_mod[1, irad] = summod / np.sum(idx_rad[1, :, irad])
                                                
for iboot in tqdm(range(nboot),desc='Bootstrapping'):
    idx_rad_shuf = my_shuffle(idx_rad[0],axis=1)

    # idx_rad_shuf = idx_rad[0][:,np.random.permutation(idx_rad[0].shape[1])]

    for irad in range(nradii-1):
        summod = np.sum(dprimesig[0,np.where(idx_rad_shuf[:, irad])[0]] !=0)
        frac_mod_boot[0, irad, iboot] = summod / np.sum(idx_rad_shuf[:, irad])

    idx_rad_shuf = my_shuffle(idx_rad[1],axis=1)
    # idx_rad_shuf = idx_rad[1][:,np.random.permutation(idx_rad[1].shape[1])]

    for irad in range(nradii-1):
        summod = np.sum(dprimesig[2,np.where(idx_rad_shuf[:, irad])[0]] !=0)
        frac_mod_boot[1, irad, iboot] = summod / np.sum(idx_rad_shuf[:, irad])

#%% Plot the bootstrapped fraction of modulated neurons as a function of distance to looped cell:
fig,ax = plt.subplots(1,2,figsize=(8*cm,4*cm),sharex=True,sharey=False)
for ialp,alp in enumerate(legendlabels):
    ax[ialp].plot(radii[:-1],frac_mod[ialp,:],color=clrs_arealabelpairs[ialp],linewidth=1,
                  marker='.',markersize=4)
    tempdata = frac_mod_boot[np.ix_([ialp],range(nradii-1),range(nboot))].squeeze()
    ax[ialp].fill_between(radii[:-1], np.percentile(tempdata,(100-95)/2,axis=1), np.percentile(tempdata,95+(100-95)/2,axis=1),
                          color='grey',alpha=0.25,edgecolor=None)
    ax[ialp].set_xticks(np.arange(0,radii[-1]+10,25))
    ax[ialp].set_xlabel('Minimum distance to looped cell (um)')
    if ialp == 0: 
        ax[ialp].set_ylabel('Fraction modulated')
    ax[ialp].set_title(legendlabels[ialp])
    ax[ialp].set_xticks(np.arange(0,250,50))
    ax[ialp].set_xlim([15,radii[-1]])
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FracMod_min_distance_V1PM_%dsessions' % nSessions)


# #%%
# #       ####   ####  #####  # #    #  ####  
# #      #    # #    # #    # # ##   # #    # 
# #      #    # #    # #    # # # #  # #      
# #      #    # #    # #####  # #  # # #  ### 
# #      #    # #    # #      # #   ## #    # 
# ######  ####   ####  #      # #    #  ####  



# # OLD!!!! 
# #%% 
# for ises in range(nSessions):
#     sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'],metric='xyz')
# celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

# # # %% Set threshold for significant correlations based on correlation value: 
# # corrsig_cells = np.full((narealabelpairs,nCells),np.nan)
# # corrsig_cells[corrdata_cells>0.25]           = 1
# # corrsig_cells[corrdata_cells<-0.25]          = -1

# #%%
# fracdata = np.full((narealabelpairs,2,nSessions),np.nan)

# for ises in range(nSessions):
#     idx_ses = np.isin(celldata['session_id'],sessions[ises].session_id)

#     for ialp,alp in enumerate(arealabelpairs):
#         idx_N = np.all((
#                         idx_ses,
#                         ~np.isnan(dprimesig[ialp,:]),
#                         # ~np.isnan(corrsig_cells[ialp,:]),
#                         celldata['nearby'],
#                         rangeresp>params['minrangeresp'],
#                         # rangeresp>0.05,
#                         ),axis=0)
#         # fracdata[ialp,0,ises] = np.sum(corrsig_cells[ialp,idx_N]==1) / np.sum(idx_N)
#         # fracdata[ialp,1,ises] = np.sum(corrsig_cells[ialp,idx_N]==-1) / np.sum(idx_N)

#         fracdata[ialp,0,ises] = np.sum(dprimesig[ialp,idx_N]==1) / np.sum(idx_N)
#         fracdata[ialp,1,ises] = np.sum(dprimesig[ialp,idx_N]==-1) / np.sum(idx_N)

#         # fracdata[ialp,1,ises] = np.sum(corrsig_cells[ialp,idx_ses]==-1) / np.sum(~np.isnan(corrsig_cells[ialp,idx_ses]))

# clrs = ['black','red']
# axtitles = np.array(['FF: +corr','FF: -corr', 'FB: +corr','FB: -corr'])

# fig,axes = plt.subplots(1,4,figsize=(12*cm,3*cm))
# ax = axes[0]
# sns.barplot(data=fracdata[:2,0,:].T,palette=clrs,ax=ax,alpha=0.3)
# sns.stripplot(data=fracdata[:2,0,:].T,palette=clrs,ax=ax)
# h,p = stats.ttest_rel(fracdata[0,0,:],fracdata[1,0,:],nan_policy='omit')
# add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[:2,0,:]), p, h=0)
# ax.set_xticklabels(arealabelpairs[:2])
# print(p)

# ax = axes[1]
# sns.barplot(data=fracdata[:2,1,:].T,palette=clrs,ax=ax,alpha=0.3)
# sns.stripplot(data=fracdata[:2,1,:].T,palette=clrs,ax=ax)
# h,p = stats.ttest_rel(fracdata[0,1,:],fracdata[1,1,:],nan_policy='omit')
# add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[:2,1,:]), p, h=0)
# ax.set_xticklabels(arealabelpairs[:2])
# print(p)

# ax = axes[2]
# sns.barplot(data=fracdata[2:,0,:].T,palette=clrs,ax=ax,alpha=0.3)
# sns.stripplot(data=fracdata[2:,0,:].T,palette=clrs,ax=ax)
# h,p = stats.ttest_rel(fracdata[2,0,:],fracdata[3,0,:],nan_policy='omit')
# add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[2:,0,:]), p, h=0)
# ax.set_xticklabels(arealabelpairs[2:])
# print(p)

# ax = axes[3]
# sns.barplot(data=fracdata[2:,1,:].T,palette=clrs,ax=ax,alpha=0.3)
# sns.stripplot(data=fracdata[2:,1,:].T,palette=clrs,ax=ax)
# h,p = stats.ttest_rel(fracdata[2,1,:],fracdata[3,1,:],nan_policy='omit')
# add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[2:,1,:]), p, h=0)
# ax.set_xticklabels(arealabelpairs[2:])
# print(p)

# for iax,ax in enumerate(axes):
#     ax.set_title(axtitles[iax])
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'FF_FB_affinemodulation_FracSig_%dsessions' % (nSessions), formats = ['png'])

# # #%%
# # for ialp,alp in enumerate(arealabelpairs):
# #     plt.scatter(fracdata[ialp,0,:],fracdata[ialp,1,:])

# #%%
# legendlabels    = ['nonlooped','looped']

# fracmat         = np.full((3,3,2,3),np.nan)
# nsigmat         = np.full((3,3,2,2),np.nan)
# ntotalmat       = np.full((3,3,2,2),np.nan)
# testmat         = np.full((3,3,2),np.nan)
# ncomparisons    = 1

# for ialp,alp in enumerate(arealabelpairs):
#     idir = ialp//2
#     icat = ialp%2
#     idx_N = np.all((
#                     rangeresp>params['minrangeresp'],
#                     celldata['nearby'],
#                     # celldata['noise_level']<maxnoiselevel,
#                     # corrsig_cells[icat,:],
#                     # corrsig_cells[ialp,:] !=1,
#                      ),axis=0)
#     for imult, mult in enumerate([1,0,-1]):
#         for iadd, add in enumerate([-1,0,1]):
#             Nsig = np.sum(np.all((
#                                 sig_params_regress[idx_N,ialp,0]==mult,
#                                 sig_params_regress[idx_N,ialp,1]==add,
#                                 ),axis=0))
#             Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
#             frac = (Nsig/Ntotal) * 100

#             nsigmat[imult,iadd,idir,icat] = Nsig
#             ntotalmat[imult,iadd,idir,icat] = Ntotal
#             fracmat[imult,iadd,idir,icat] = frac
# fracmat[:,:,:,2] = fracmat[:,:,:,1] - fracmat[:,:,:,0]
# # fracmat[:,:,:,2] = fracmat[:,:,:,1] - fracmat[:,:,:,0]

# for i in range(2):
#     for imult, mult in enumerate([1,0,-1]):
#         for iadd, add in enumerate([-1,0,1]):
#             data = np.array([[nsigmat[imult,iadd,i,0], ntotalmat[imult,iadd,i,0]-nsigmat[imult,iadd,i,0]],
#                             [nsigmat[imult,iadd,i,1], ntotalmat[imult,iadd,i,1]-nsigmat[imult,iadd,i,1]]])
#             if np.all(data[:,0]==0): 
#                 continue
#             testmat[imult,iadd,i] = stats.chi2_contingency(data)[1]  # p-value
# testmat = testmat * ncomparisons  #bonferroni correction

# fig,axes = plt.subplots(2,3,figsize=(9,6))
# for idir in range(2):
#     for icat in range(3):
#         ax = axes[idir,icat]
#         if icat < 2:
#             vmin,vmax = 0,15
#             # cmap = 'Purples'
#             cmap = 'viridis'
#             # cmap = 'magma'
#             # cmap = 'Greens'
#         else:
#             vmin,vmax = -5,5
#             cmap = 'bwr'
#             # cmap = 'PiYG'
#         im = ax.imshow(fracmat[:,:,idir,icat],vmin=vmin,vmax=vmax,cmap=cmap)

#         ax.set_xticks([0,1,2],['Sub','None','Add'])
#         # ax.set_yticks([0,1,2],['Div','None','Mult'])
#         ax.set_yticks([0,1,2],['Mult','None','Div'])
#         ax.set_xlabel('Addition')
#         if icat == 0:
#             ax.set_ylabel('Multiplicative')
#         ax.set_title(legendlabels[icat] if icat < 2 else 'Diff (%s-%s)' % (legendlabels[1],legendlabels[0]))
#         for i in range(3):
#             for j in range(3):
#                 if icat != 2:
#                     ax.text(j,i,'%2.1f%%' % fracmat[i,j,idir,icat],ha='center',va='center',color='white' if fracmat[i,j,idir,icat]<20 else 'black')
#                 else: 
#                     ax.text(j,i,'%s%2.1f%%\n%s' % ('+' if fracmat[i,j,idir,icat]>0 else '',fracmat[i,j,idir,icat],get_sig_asterisks(testmat[i,j,idir])),
#                             ha='center',va='center',color='black')
#     fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04,label='% sign. cells')
# plt.tight_layout()
# # sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'Affine_sig_mod_FF_FB_heatmap_%dsessions' % (nSessions))



# #%% Show mean tuned responses and modulation between looped and nonlooped neurons: 
# fig,axes = plt.subplots(2,4,figsize=(10,3),sharex=True,sharey=True)
# clrs = ['black','red']
# axtitles = np.array([['FF PMunl: -corr','FF PMlab: -corr', 'FB V1unl: -corr','FB V1lab: -corr'],
#                      ['FF PMunl: +corr','FF PMlab: +corr', 'FB V1unl: +corr','FB V1lab: +corr']])

# for ialp,alp in enumerate(arealabelpairs):
#     for isign,sign in enumerate([-1,1]):
#         ax = axes[isign,ialp]
#         idx_N = np.all((
#                         dprimesig[ialp,:]==sign,
#                         # corrsig_cells[ialp,:]==sign,
#                         celldata['nearby'],
#                         # celldata['noise_level']<maxnoiselevel,
#                         rangeresp>params['minrangeresp'],
#                         ),axis=0)
#         meandata = np.nanmean(mean_resp_split_aligned[ialp,:,0,idx_N],axis=0)
#         ax.plot(oris,meandata,color=clrs[0],alpha=1)
#         meandata = np.nanmean(mean_resp_split_aligned[ialp,:,1,idx_N],axis=0)
#         ax.plot(oris,meandata,color=clrs[1],alpha=1)
#         ax.set_title(axtitles[isign,ialp])
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'FF_FB_Looped_Correlations_bootstrapped_%dsessions' % (nSessions), formats = ['png'])









# #%% Bootstrapped comparison of correlations and significant correlations with other area: 

# # For each bootstrap, for each labeled cell a random nonlabeled cell that is nearby is sampled. 
# # This results in an equal number of nonlabeled cells in a paired way. 
# # The distribution of correlations is compared to the loop correlation distribution.
# # The fraction of significantly positive and negative as well. 
# radius          = 50

# nboots          = 10

# loopfrac        = np.full((2,3),np.nan) # FF vs FB, +corr vs -corr vs modulated
# loopmean        = np.full((2),np.nan) # FF vs FB, +corr vs -corr
# loopmean_abs    = np.full((2),np.nan) # FF vs FB, +corr vs -corr

# bootfrac        = np.full((2,3,nboots),np.nan) # FF vs FB, +corr vs -corr
# bootmean        = np.full((2,nboots),np.nan) # FF vs FB, +corr vs -corr
# bootmean_abs    = np.full((2,nboots),np.nan) # FF vs FB, +corr vs -corr

# binedges        = np.linspace(-1,1,50)
# nhistbins       = len(binedges)-1
# loophist        = np.full((2,nhistbins),np.nan) # FF vs FB, +corr vs -corr
# boothist        = np.full((2,nhistbins,nboots),np.nan) # FF vs FB, +corr vs -corr

# idx_N           = np.all((
#                     # celldata['noise_level']<params['maxnoiselevel'],
#                     rangeresp>params['minrangeresp'],
#                     ),axis=0)

# idx_PMlab       = np.where(celldata['arealayerlabel'] == 'PMlabL2/3')[0]
# idx_V1lab       = np.where(celldata['arealayerlabel'] == 'V1labL2/3')[0]
# idx_PMlab_hasnearby = np.full(len(idx_PMlab),False)
# idx_V1lab_hasnearby = np.full(len(idx_V1lab),False)

# for iboot in tqdm(range(nboots),total=nboots,desc='Bootstrapping'):
#     idx_PMlab_nearby = np.full(len(idx_PMlab),np.nan)
#     for iN,N in enumerate(idx_PMlab):
#         #get index of which session this labeled cell comes from:
#         ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
#         #get index of all cells in this session
#         idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
#         #get index of labeled cell in this session
#         idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
#         #get index of all unlabeled cells in this session that are nearby this particular labeled cell
#         idx_nearby_ses = np.where(np.all((np.squeeze(sessions[ises].distmat_xyz[idx_N_ses,:]<radius),
#                                                  rangeresp[idx_ses]>params['minrangeresp'],
#                                                  sessions[ises].celldata['redcell']==0,
#                                                 #  sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                                  ),axis=0))[0]
        
#         #Convert this index to the index in the whole dataset
#         idx_nearby = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
#         if len(idx_nearby) > 0: #pick a random one from the selected nearby cells
#             idx_PMlab_nearby[iN] = np.random.choice(idx_nearby,1)
#             idx_PMlab_hasnearby[iN] = True
#     if iboot == 0: 
#         print('PMlab: %d/%d' % (np.sum(~np.isnan(idx_PMlab_nearby)),len(idx_PMlab)))
#     idx_PMlab_nearby = idx_PMlab_nearby[~np.isnan(idx_PMlab_nearby)].astype(int) #remove nans

#     bootfrac[0,0,iboot] = np.sum(corrsig_cells[0,idx_PMlab_nearby]==1) / len(idx_PMlab_nearby) #compute fraction of sig pos for this boot
#     bootfrac[0,1,iboot] = np.sum(corrsig_cells[0,idx_PMlab_nearby]==-1) / len(idx_PMlab_nearby)
    
#     histcounts = np.histogram(corrdata_cells[0,idx_PMlab_nearby],bins=binedges)[0]
#     boothist[0,:,iboot] = np.cumsum(histcounts)/np.sum(histcounts)
#     bootmean[0,iboot] = np.nanmean(corrdata_cells[0,idx_PMlab_nearby])
#     bootmean_abs[0,iboot] = np.nanmean(np.abs(corrdata_cells[0,idx_PMlab_nearby]))

#     idx_V1lab_nearby = np.full(len(idx_V1lab),np.nan)
#     for iN,N in enumerate(idx_V1lab):
#         ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0]
#         idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
#         idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
        
#         idx_nearby_ses = np.where(np.all((np.squeeze(sessions[ises].distmat_xyz[idx_N_ses,:]<radius),
#                                                  rangeresp[idx_ses]>params['minrangeresp'],
#                                                  sessions[ises].celldata['redcell']==0,
#                                                 #  sessions[ises].celldata['noise_level']<maxnoiselevel,
#                                                  ),axis=0))[0]
#         idx_nearby = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]

#         if len(idx_nearby) > 0:
#             idx_V1lab_nearby[iN] = np.random.choice(idx_nearby,1)
#             idx_V1lab_hasnearby[iN] = True
#     if iboot == 0: 
#         print('V1lab: %d/%d' % (np.sum(~np.isnan(idx_V1lab_nearby)),len(idx_V1lab_nearby)))
#     idx_V1lab_nearby = idx_V1lab_nearby[~np.isnan(idx_V1lab_nearby)].astype(int)

#     bootfrac[1,0,iboot] = np.sum(corrsig_cells[2,idx_V1lab_nearby]==1) / len(idx_V1lab_nearby)
#     bootfrac[1,1,iboot] = np.sum(corrsig_cells[2,idx_V1lab_nearby]==-1) / len(idx_V1lab_nearby)
#     bootfrac[:,2,:] = (bootfrac[:,0,:]+bootfrac[:,1,:])  

#     histcounts = np.histogram(corrdata_cells[2,idx_V1lab_nearby],bins=binedges)[0]
#     boothist[1,:,iboot] = np.cumsum(histcounts)/np.sum(histcounts)
#     bootmean[1,iboot] = np.nanmean(corrdata_cells[2,idx_V1lab_nearby])
#     bootmean_abs[1,iboot] = np.nanmean(np.abs(corrdata_cells[2,idx_V1lab_nearby]))

# #%% Now calculated actual looped data:
# idx_N           = np.all((
#                     rangeresp>params['minrangeresp'],
#                     # np.logical_or(np.isin(range(nCells),idx_V1lab[idx_V1lab_hasnearby]),
#                                 #   np.isin(range(nCells),idx_PMlab[idx_PMlab_hasnearby])),
#                     ),axis=0)
# # print(np.sum(idx_N))
# loopfrac[0,0]   = np.sum(corrsig_cells[1,idx_N]==1) / np.sum(~np.isnan(corrsig_cells[1,idx_N]))
# loopfrac[0,1]   = np.sum(corrsig_cells[1,idx_N]==-1) / np.sum(~np.isnan(corrsig_cells[1,idx_N]))
# loopfrac[1,0]   = np.sum(corrsig_cells[3,idx_N]==1) / np.sum(~np.isnan(corrsig_cells[3,idx_N]))
# loopfrac[1,1]   = np.sum(corrsig_cells[3,idx_N]==-1) / np.sum(~np.isnan(corrsig_cells[3,idx_N]))
# loopfrac[:,2]   = loopfrac[:,0] + loopfrac[:,1]

# loopmean[0]     = np.nanmean(corrdata_cells[1,idx_N])
# loopmean[1]     = np.nanmean(corrdata_cells[3,idx_N])
# loopmean_abs[0] = np.nanmean(np.abs(corrdata_cells[1,idx_N]))
# loopmean_abs[1] = np.nanmean(np.abs(corrdata_cells[3,idx_N]))

# histcounts      = np.histogram(corrdata_cells[1,idx_N],bins=binedges)[0]
# loophist[0,:]   = np.cumsum(histcounts)/np.sum(histcounts)
# histcounts      = np.histogram(corrdata_cells[3,idx_N],bins=binedges)[0]
# loophist[1,:]   = np.cumsum(histcounts)/np.sum(histcounts)

# #%% 
# legendlabels        = ['FF','FB']
# axisbuffer          = 0.025
# lw                  = 2

# # subplotlabels = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.'])
# # loopdata_subplots = np.stack((loopmean,loopmean_abs,loopfrac[:,0],loopfrac[:,1]),axis=1)
# # bootdata_subplots = np.stack((bootmean,bootmean_abs,bootfrac[:,0],bootfrac[:,1]),axis=1)
# subplotlabels = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.','Frac. Mod.'])
# loopdata_subplots = np.stack((loopmean,loopmean_abs,loopfrac[:,0],loopfrac[:,1],loopfrac[:,2]),axis=1)
# bootdata_subplots = np.stack((bootmean,bootmean_abs,bootfrac[:,0],bootfrac[:,1],bootfrac[:,2]),axis=1)
# nmetrics = len(subplotlabels)

# fig,axes = plt.subplots(2,nmetrics+1,figsize=(nmetrics*2.2,4))

# for ialp in range(2):
#     axes[ialp,0].plot(binedges[:-1],loophist[ialp,:],color=clrs_arealabelpairs[ialp])
    
#     shaded_error(binedges[:-1],np.nanmean(boothist[ialp,:,:],axis=1),np.nanstd(boothist[ialp,:,:],axis=1),
#                     ax=axes[ialp,0],color='grey')
#     axes[ialp,0].set_xlim([binedges[np.where(loophist[ialp,:]>0)[0][0]],binedges[np.where(loophist[ialp,:]>0.999)[0][0]]])
#     axes[ialp,0].set_ylim([0,1])
#     axes[ialp,0].set_ylabel(legendlabels[ialp],fontsize=15,fontweight='bold',color=clrs_arealabelpairs[ialp])
#     if ialp == 0:
#         axes[ialp,0].set_title('Corr. coeff.')
    
#     for imetric in range(nmetrics):
#         axidx = imetric+1
#         axes[ialp,axidx].axvline(loopdata_subplots[ialp,imetric],color=clrs_arealabelpairs[ialp],linewidth=lw)
#         sns.histplot(bootdata_subplots[ialp,imetric,:],ax=axes[ialp,axidx],bins=np.linspace(-.1,1,500),element='step',stat='probability',color='grey')
#         axes[ialp,axidx].set_xlim([np.percentile(bootdata_subplots[ialp,imetric],0)-axisbuffer,
#                                    np.percentile(bootdata_subplots[ialp,imetric],100)+axisbuffer])
#         pval = np.sum(bootdata_subplots[ialp,imetric,:]>loopdata_subplots[ialp,imetric])/len(bootdata_subplots[ialp,imetric,:])
#         axes[ialp,axidx].text(loopdata_subplots[ialp,imetric],0.1,get_sig_asterisks(np.min([pval,1-pval]),return_ns=True),fontsize=16,color=clrs_arealabelpairs[ialp])
#         if ialp == 0:
#             axes[ialp,axidx].set_title(subplotlabels[imetric])

# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# # my_savefig(fig,savedir,'FF_FB_Looped_Correlations_bootstrapped_%dsessions' % (nSessions), formats = ['png'])

# #%% 




