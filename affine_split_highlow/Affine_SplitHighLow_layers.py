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
from scipy.stats import linregress,binned_statistic,pearsonr,spearmanr,ks_2samp
from scipy import stats
from statsmodels.formula.api import ols

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper,ori_remapping
from utils.gain_lib import * 
from utils.pair_lib import compute_pairwise_anatomical_distance,value_matching,filter_nearlabeled
from utils.plot_lib import * #get all the fixed color schemes

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\SplitTrials\\Layers')

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

#%% Compute Tuning Metrics (gOSI, gDSI etc.)
sessions = ori_remapping(sessions)
sessions = compute_tuning_wrapper(sessions)

#%% Identify cells near labeled cells
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'])

#%% Get concatenated data:
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)


#%% 
#          #    #     # ####### ######   #####  
#         # #    #   #  #       #     # #     # 
#        #   #    # #   #       #     # #       
#       #     #    #    #####   ######   #####  
#       #######    #    #       #   #         # 
#       #     #    #    #       #    #  #     # 
####### #     #    #    ####### #     #  #####  

#%%

####### ######     ####### ######  ###  #####  ### #     # 
#       #     #    #     # #     #  #  #     #  #  ##    # 
#       #     #    #     # #     #  #  #        #  # #   # 
#####   ######     #     # ######   #  #  ####  #  #  #  # 
#       #     #    #     # #   #    #  #     #  #  #   # # 
#       #     #    #     # #    #   #  #     #  #  #    ## 
#       ######     ####### #     # ###  #####  ### #     # 

#%% Check whether feedback from PM to V1 modulates V1 depending on the laminar origin of the feedback

arealabelpairs          = [
                            'PMlabL2/3-PMunlL2/3-V1unlL2/3',
                            'PMlabL5-PMunlL5-V1unlL2/3',
                            ]
celldatafields           = ['arealayerlabel','arealayerlabel','arealayerlabel']
narealabelpairs         = len(arealabelpairs)
legendlabels            = np.array(['PML23->V1','PML5->V1'])
# clrs_arealabelpairs = get_clr_arealabelpairs(arealabelpairs)
# clrs_arealabelpairs = ['green','green','purple','purple']
clrs_arealabelpairs = sns.color_palette('dark',4)[:2]
clrs_arealabels_low_high = get_clr_area_low_high()
direction = 'FB'

#%% 
####### #######    #######    #    ######   #####  ####### #######    #          #    #     # ####### ######  
#       #             #      # #   #     # #     # #          #       #         # #    #   #  #       #     # 
#       #             #     #   #  #     # #       #          #       #        #   #    # #   #       #     # 
#####   #####         #    #     # ######  #  #### #####      #       #       #     #    #    #####   ######  
#       #             #    ####### #   #   #     # #          #       #       #######    #    #       #   #   
#       #             #    #     # #    #  #     # #          #       #       #     #    #    #       #    #  
#       #             #    #     # #     #  #####  #######    #       ####### #     #    #    ####### #     # 

#%% Check whether feedforward from V1 to PM modulates PM depending on the target layer
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'V1lab-V1unl-PMunlL5',
                    ]
celldatafields           = ['arealabel','arealabel','arealayerlabel']

legendlabels            = np.array(['V1->PML23','V1->PML5'])

# clrs_arealabelpairs = ['green','green','purple','purple']
# clrs_arealabelpairs = get_clr_arealabelpairs(arealabelpairs)
clrs_arealabelpairs = sns.color_palette('dark',4)[2:]
direction = 'FF'


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

#Correlation output:
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(sessions[ises].celldata[celldatafields[0]] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata[celldatafields[1]] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata[celldatafields[2]] == alp.split('-')[2])[0]

        # subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
        # idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
        # idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

        if len(idx_N1) < params['minnneurons'] or len(idx_N2) < params['minnneurons']:
            print('Not enough neurons (%d) in the target area session %d, area pair %s' % (len(idx_N1),ises,alp))
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

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<params['alpha_crossrate']) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

# print(np.sum(~np.isnan(corrdata_cells[0,:])))
# print(np.sum(~np.isnan(corrdata_cells[1,:])))

#%% Store results in celldata
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
for ises in range(nSessions):
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    sessions[ises].celldata['correlation_PML23']   = corrdata_cells[0,idx_ses]
    sessions[ises].celldata['correlation_PML5']   = corrdata_cells[1,idx_ses]
    sessions[ises].celldata['sigmod_FBPML23']       = corrsig_cells[0,idx_ses]
    sessions[ises].celldata['sigmod_FBPML5']       = corrsig_cells[1,idx_ses]
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%% Feedback modulated cells:
ises = 2
iplane = 3
cellfield = 'correlation_PML23'
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_correlation_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')

cellfield = 'correlation_PML5'
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_correlation_V1L5_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')

#%%
ises = 7
iplane =4
cellfield = 'correlation_PML23'
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_correlation_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')

cellfield = 'correlation_PML5'
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_correlation_V1L5_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')


#%% Are neurons similarly modulated by FB from L2/3 or L5?

fig,axes = plt.subplots(1,1,figsize=(4*cm,3.7*cm))
# ax = axes[ialp]
ax = axes
# ax.scatter(corrdata_cells[0,:],corrdata_cells[1,:],color='#666666',marker='.',s=3)
sns.regplot(x=corrdata_cells[0,:],y=corrdata_cells[1,:],
            scatter_kws={'s':2, 'alpha':0.5,'marker':'.','edgecolor':'none','color':'#666666'},
            line_kws={'color':"#313DCA",'lw':1,'alpha':1},ci=95,ax=ax)
ax_nticks(ax,4)
ax.set_xlabel('Correlation PM-L23')
ax.set_ylabel('Correlation PM-L5')
ax.set_xticks(np.arange(-1,1.1,0.25))
ax.set_yticks(np.arange(-1,1.1,0.25))
ax.set_ylim([-.25,0.6])
ax.set_xlim([-.25,0.6])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Correlation_%s_DiffLayers_%dsessions' % (direction,nSessions))

#%% Show pie chart of significant correlation for feedforward and feedback:
sigmat = np.empty((3,narealabelpairs))
signorder = [-1,0,1]
signlabels = ['Neg','None','Pos']
signorder = [-1,1,0]
signlabels = ['Neg','Pos','None']   
clrs_signs = ['#0033B3','#E66804','#808080']
for ialp,alp in enumerate(arealabelpairs):
    for isign,sign in enumerate(signorder):
        idx_N = ~np.isnan(corrdata_cells[ialp,:])

        sigmat[isign,ialp] = np.sum(corrsig_cells[ialp,idx_N]==sign) / np.sum(idx_N)

#Make the figure:
fig,axes = plt.subplots(1,2,figsize=(7*cm,4*cm))
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    ax.pie([sigmat[0,ialp],sigmat[1,ialp],sigmat[2,ialp]],labels=signlabels,colors=clrs_signs,autopct='%1.1f%%',
            startangle=90,counterclock=False,wedgeprops = {'linewidth': 0.8, 'edgecolor': 'black', 'alpha': 0.7},
            textprops={'fontsize': 6})
    ax.set_title('%s\nn=%d' % (legendlabels[ialp],np.sum(~np.isnan(corrdata_cells[ialp,:]))))
    # ax.text(-0.1,0.1,'n=%d' % np.sum(~np.isnan(corrdata_cells[ialp,:])),transform=ax.transAxes,ha='center',va='center',
    #         fontsize=6)
my_savefig(fig,savedir,'Corr_%s_Sign_PieCharts_%dsessions' % (direction,nSessions))

#%% Fraction of significant multiplicative and additively modulated cells:
sign = 1
fig,axes = plt.subplots(1,1,figsize=(6*cm,4*cm))
ax = axes
idx_N =  rangeresp>params['minrangeresp']
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
ax.text(0.6, 0.9, '%s (n=%d)' % (legendlabels[0],np.sum(corrsig_cells[0,idx_N]==sign)), color=clrs_arealabelpairs[0],transform=ax.transAxes)
ax.text(0.6, 0.8, '%s (n=%d)' % (legendlabels[1],np.sum(corrsig_cells[1,idx_N]==sign)), color=clrs_arealabelpairs[1],transform=ax.transAxes)
ax.set_xticks(np.arange(3)*2+0.5,['Mult','Add','Both'])
ax.set_ylabel('Fraction of + correlated cells')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'%s_affinemodulation_sig_posmod_Layers_%dsessions' % (direction,nSessions))

#%% Fraction of significant multiplicative and additively modulated cells:
sign = -1
fig,axes = plt.subplots(1,1,figsize=(6*cm,4*cm))
ax = axes
idx_N =  rangeresp>params['minrangeresp']
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
    
ax.text(0.6, 0.9, '%s (n=%d)' % (legendlabels[0],np.sum(corrsig_cells[0,idx_N]==sign)), color=clrs_arealabelpairs[0],transform=ax.transAxes)
ax.text(0.6, 0.8, '%s (n=%d)' % (legendlabels[1],np.sum(corrsig_cells[1,idx_N]==sign)), color=clrs_arealabelpairs[1],transform=ax.transAxes)

ax.set_xticks(np.arange(3)*2+0.5,['Div','Sub','Both'])
ax.set_ylabel('Fraction of - correlated cells')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'%s_affinemodulation_sig_negmod_Layers_%dsessions' % (direction,nSessions))

# #%%
# fracmat     = np.full((3,3,narealabelpairs+1),np.nan)
# nsigmat     = np.full((3,3,narealabelpairs),np.nan)
# ntotalmat   = np.full((3,3,narealabelpairs),np.nan)
# testmat     = np.full((3,3),np.nan)
# ncomparisons = 9
# for ialp,alp in enumerate(arealabelpairs):
#     # for imult, mult in enumerate([-1,0,1]):
#     for imult, mult in enumerate([1,0,-1]):
#         for iadd, add in enumerate([-1,0,1]):
#             idx_N =  np.all((
#                 rangeresp>0.05,
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
# fracmat[:,:,2] = fracmat[:,:,1] - fracmat[:,:,0]

# for imult, mult in enumerate([1,0,-1]):
#     for iadd, add in enumerate([-1,0,1]):
#         data = np.array([[nsigmat[imult,iadd,0], ntotalmat[imult,iadd,0]-nsigmat[imult,iadd,0]],
#                          [nsigmat[imult,iadd,1], ntotalmat[imult,iadd,1]-nsigmat[imult,iadd,1]]])
#         testmat[imult,iadd] = stats.chi2_contingency(data)[1]  # p-value
# testmat = testmat * ncomparisons  #bonferroni correction

# fig,axes = plt.subplots(1,3,figsize=(9,3))
# for ialp in range(narealabelpairs+1):
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
#     ax.set_title(legendlabels[ialp] if ialp < narealabelpairs else 'Diff (FB-FF)')
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
# my_savefig(fig,savedir,'Affine_sig_mod_FF_FB_heatmap_layers_%dsessions' % (nSessions))


#%% 
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
        idx_N = np.all((
                rangeresp>params['minrangeresp'],
                corrsig_cells[ialp,:]==1,
                ),axis=0)
        sns.histplot(data=params_regress[idx_N,ialp,iparam],element='step',
                     color=clrs_arealabelpairs[ialp],
                     alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
        handles.append(ax.plot(np.nanmean(params_regress[idx_N,ialp,iparam]),0.95,markersize=6,
                color=clrs_arealabelpairs[ialp],marker='v')[0])
        ncells = np.sum(~np.isnan(params_regress[idx_N,ialp,iparam]))

        ax.text(0.4, 0.1+ialp*0.1, '%s (n=%d)' % (legendlabels[ialp],ncells), 
                transform=ax.transAxes,fontsize=5,color=clrs_arealabelpairs[ialp])
    
    idx_N = np.all((
                rangeresp>params['minrangeresp'],
                ),axis=0)
    
    h,p = stats.ttest_ind(params_regress[idx_N,0,iparam],
                            params_regress[idx_N,1,iparam],nan_policy='omit')
    p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
    # ax.text(0.6, 0.15, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes,fontsize=9)
    ax.text(0.45, 0.5, '%s' % (get_sig_asterisks(p,return_ns=True)),
            transform=ax.transAxes)
    ax.set_yticks([0,0.5,1.0])
    ax.set_xlim(xlims)
    ax.set_ylabel('Cumulative fraction of cells')

    # ax.legend(handles,legendlabels,frameon=False,loc='center right')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=2)

my_savefig(fig,savedir,'%s_affinemodulation_layers_posmod_cumhistcoefs_%dGRsessions' % (direction,nSessions))

#%% Is the effect similar for the two areas, but 
# just dependent on a difference in activity levels?
# Is modulation more multiplicative for larger activity levels, stimuli with 

ci              = 95 #bootstrapped confidence interval
nboots          = 250 #number of bootstrap samples
percspacing     = 5 #bins chosen to have approx equal number of points
percentiles     = np.arange(0,100+percspacing,percspacing)
percentiles[percentiles==100] = 99.75 #avoid issues with max value
bins            = np.nanpercentile(mean_resp_split,percentiles)
bins            = bins[bins>0] #remove duplicate bins at 0
bincenters      = (bins[:-1]+bins[1:])/2 #get bin centers

resp_mod = mean_resp_split[:,:,1,:] - mean_resp_split[:,:,0,:]

fig,axes = plt.subplots(1,3,figsize=(12*cm,4*cm),sharex=True,sharey=False)
ax = axes[0]
handles = []
bootdata = np.full((narealabelpairs,len(bins)-1,nboots),np.nan)
bootci = np.full((narealabelpairs,2,len(bins)-1),np.nan)
for ialp,alp in enumerate(arealabelpairs):
    idx_N =  rangeresp>params['minrangeresp']
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
# ax.legend(handles,legendlabels,frameon=False,loc='best')
ax.set_ylabel('Modulation')
ax.axhline(0,color='grey',ls='--',linewidth=1)
ax.set_xlim([0,bincenters[-1]*1.01])
ax_nticks(ax,3)
for ibin in range(len(bincenters)):
    if bootci[0,0,ibin] > bootci[1,1,ibin] or bootci[0,1,ibin] < bootci[1,0,ibin]:
        # ax.plot(bincenters[ibin],-0.001,'k*',markersize=6)
        ax.plot(bincenters[ibin],0.01,'k*',markersize=3)

ax = axes[1]
for ialp,alp in enumerate(arealabelpairs):
    idx_N = np.all((
                rangeresp>params['minrangeresp'],
                # corrsig_cells[ialp,:]==1,
                sig_params_regress[:,ialp,0]==1,
                sig_params_regress[:,ialp,1]!=1,
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
ax.legend(handles,legendlabels,frameon=False,loc='best')
ax.set_xlim([0,bincenters[-1]])
ax.set_ylim([0,.1])
ax.set_xlabel('Activity')
ax.axhline(0,color='grey',ls='--',linewidth=1)
ax_nticks(ax,3)
for ibin in range(len(bincenters)):
    if bootci[0,0,ibin] > bootci[1,1,ibin] or bootci[0,1,ibin] < bootci[1,0,ibin]:
        # ax.plot(bincenters[ibin],-0.001,'k*',markersize=6)
        ax.plot(bincenters[ibin],0.01,'k*',markersize=6)

ax = axes[2]
for ialp,alp in enumerate(arealabelpairs):
    idx_N = np.all((          
                rangeresp>params['minrangeresp'],
                sig_params_regress[:,ialp,0]!=1,
                sig_params_regress[:,ialp,1]==1,
                # corrsig_cells[ialp,:]==-1,
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
ax.axhline(0,color='grey',ls='--',linewidth=1)

ax_nticks(ax,3)
ax.set_ylim([0,.1])
ax_nticks(ax,3)
for ibin in range(len(bincenters)):
    if bootci[0,0,ibin] > bootci[1,1,ibin] or bootci[0,1,ibin] < bootci[1,0,ibin]:
        # ax.plot(bincenters[ibin],-0.001,'k*',markersize=6)
        ax.plot(bincenters[ibin],0.01,'k*',markersize=6)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'%s_Modulation_vs_Activity_layers_%dGRsessions' % (direction,nSessions))



#%% 

   #    ####### ####### ### #     # #######     #####  #     #    #    ######     #     #####  
  # #   #       #        #  ##    # #          #     # #     #   # #   #     #   # #   #     # 
 #   #  #       #        #  # #   # #          #       #     #  #   #  #     #  #   #  #       
#     # #####   #####    #  #  #  # #####      #       ####### #     # ######  #     # #       
####### #       #        #  #   # # #          #       #     # ####### #   #   ####### #       
#     # #       #        #  #    ## #          #     # #     # #     # #    #  #     # #     # 
#     # #       #       ### #     # #######     #####  #     # #     # #     # #     #  #####  

#%% Show some example neurons:

# #%% use 
# ialp = 0
# # ialp = 1
# legendlabels        = ['FF','FB']

# #%% Get good multiplicatively modulated cells:
# #mutliplicative: 
# idx_examples = np.all((params_regress[:,ialp,0]>np.nanpercentile(params_regress[:,ialp,0],90),
#                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                         rangeresp>params['minrangeresp'],
#                        ),axis=0)
# # example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

# #%% Get good divisive modulated cells:
# idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
#                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                         rangeresp>params['minrangeresp'],
#                        ),axis=0)

# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

# #%% Get good additively modulated cells: 
# #additive:
# idx_examples = np.all((
#                         # params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
#                        params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],70),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],75),
#                         rangeresp>params['minrangeresp'],
#                        ),axis=0)
# # example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

# idx_examples = np.all((
#                        params_regress[:,ialp,0]>0.9,#slope within reasonable range of 1
#                        params_regress[:,ialp,0]<1.1,   
#                         # params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
#                        params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],85),
#                         # params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                         rangeresp>params['minrangeresp'],
#                        ),axis=0)
# # example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

# #%% Get good subtractive modulated cells: 
# idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],70),
#                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],25),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                        ),axis=0)
# # example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

# #%% 

# #%% Plot two example neurons, one FF and one FB, with tuning curve and scatter side by side
# example_cells = [
#                     'LPE11086_2024_01_10_5_0048', #FF Additive          #paper FF example 1
#                     'LPE11086_2024_01_10_4_0096', #FF additive          #paper FF example 2

#                     'LPE11086_2024_01_10_2_0046', #FB Multiplicative    #paper FB example 1
#                     'LPE10885_2023_10_12_5_0110', #FB Multiplicative    #paper FB example 2
#                     ]

# #%% List of additional example FF cells:
# example_cells = [
#                     'LPE11086_2024_01_05_4_0020', #FF additive 
#                     'LPE11086_2024_01_05_5_0030', #FF additive
#                     'LPE11086_2024_01_05_5_0169', #FF multiplicative
#                     # 'LPE09665_2023_03_21_7_0011', #FF divisive
#                     # 'LPE11086_2024_01_05_6_0103', #FF additive
#                     # 'LPE09830_2023_04_10_5_0065', #FF additive
#                     # 'LPE11086_2024_01_05_4_0002', #FF additive
#                     # 'LPE11086_2024_01_05_4_0235', #FF additive
#                     # 'LPE11086_2024_01_05_4_0075', #FF additive
#                     # 'LPE11086_2024_01_10_4_0017', #FF additive
#                     # 'LPE11086_2024_01_10_5_0048', #FF additive
#                     # 'LPE11086_2024_01_05_6_0304', #FF additive
#                     # 'LPE11086_2024_01_10_4_0055', #FF additive
#                     # 'LPE11086_2024_01_10_4_0017', #FF additive
#                     'LPE11086_2024_01_10_4_0014', #FF additive
#                     'LPE10885_2023_10_19_0_0037', #FF additive
#                     'LPE11086_2024_01_05_4_0053', #FF additive
#                     'LPE11086_2024_01_10_4_0096', #FF additive
#                     # 'LPE11086_2024_01_05_4_0040', #FF additive
#                     # 'LPE10919_2023_11_06_0_0322', #FF subtractive/divisive
#                     ]

# #%%
# example_cells = [
#                     'LPE12223_2024_06_10_1_0051', #FB multiplicative  #paper FB example 2
#                     'LPE11086_2024_01_10_2_0046', #FB multiplicative
#                     'LPE10885_2023_10_12_6_0014', #FB Multiplicative
#                     'LPE10885_2023_10_12_5_0110', #FB Multiplicative
#                     # 'LPE11086_2024_01_05_0_0030', #FB additive
#                     # 'LPE11086_2024_01_10_3_0108', #FB multiplicative     
#                     # 'LPE10885_2023_10_12_5_0036', #FB Multiplicative
#                     # 'LPE10885_2023_10_12_4_0140', #FB Multiplicative
#                     # 'LPE11086_2024_01_10_0_0009', #FB additive
#                     # 'LPE10885_2023_10_23_1_0276', #FB divisive
#                     # 'LPE10919_2023_11_06_5_0304', #FB divisive
#                     # 'LPE11086_2024_01_10_0_0143', #FB additive
#                 ]

# #%% Plot in two ways:
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]
# for example_cell in example_cells:
#     idx_N = np.where(celldata['cell_id']==example_cell)[0][0]
#     ialp = np.where(~np.isnan(mean_resp_split[:,0,0,idx_N]))[0][0]
#     ustim = np.unique(sessions[ises].trialdata['Orientation'])
#     x = mean_resp_split[ialp,:,0,idx_N]
#     y = mean_resp_split[ialp,:,1,idx_N]
#     xerror = error_resp_split[ialp,:,0,idx_N]
#     yerror = error_resp_split[ialp,:,1,idx_N]
    
#     # clrs_stimuli    = sns.color_palette('viridis',8)
#     fig,axes = plt.subplots(1,2,figsize=(7*cm,3.5*cm))

#     ax = axes[0]
#     ax.scatter(ustim,x,color=clrs_arealabels_low_high[ialp,0],s=10)
#     ax.plot(ustim,x,color=clrs_arealabels_low_high[ialp,0],linestyle='-')
#     # ax.errorbar(ustim,x,yerr=xerror,color='k',ls='None',linewidth=1)
#     ax.errorbar(ustim,x,yerr=xerror,color=clrs_arealabels_low_high[ialp,0],ls='None')

#     ax.scatter(ustim,y,color=clrs_arealabels_low_high[ialp,1],s=10)
#     ax.plot(ustim,y,color=clrs_arealabels_low_high[ialp,1],linestyle='-')
#     # ax.errorbar(ustim,y,yerr=yerror,color='k',ls='None',linewidth=1)
#     ax.errorbar(ustim,y,yerr=yerror,color=clrs_arealabels_low_high[ialp,1],ls='None')
#     ax.set_xlabel('stimulus direction (deg)')
#     ax.set_ylabel('response')
#     ax_nticks(ax,4)
#     ax.set_xticks([0,90,180,270,360])
#     ax.tick_params(axis='both', which='major')

#     ax = axes[1]
#     ax.scatter(x,y,color='#666666',s=5)
#     # ax.errorbar(x,y,xerr=xerror,yerr=yerror,color='k',ls='None')
#     b = linregress(x, y)
#     xp = np.linspace(np.percentile(x,0),np.percentile(x,100)*1.1,100)
#     ax.plot(xp,b[0]*xp+b[1],color=clrs_arealabelpairs[ialp],linestyle='-',linewidth=1.5)

#     # ax.text(0.5,0.05,'Slope: %1.2f\nOffest: %1.2f'%(b[0],my_ceil(b[1],precision=2)),
#     ax.text(0.5,0.05,'Slope: %1.2f\nOffest: %1.2f'%(b[0],round(b[1],2)),
#                     transform=ax.transAxes,color='k',fontsize=6)
#     ax.tick_params(axis='both', which='major')

#     ax.plot([0,1],[0,1],color='grey',ls='--')
#     ax.set_xlim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
#     ax.set_ylim([np.nanmin([x,y]),np.nanmax([x,y])*1.1])
#     ax.set_ylabel('%s high' % legendlabels[ialp],color=clrs_arealabels_low_high[ialp,1])
#     ax.set_xlabel('%s low' % legendlabels[ialp],color=clrs_arealabels_low_high[ialp,0])
#     # ax.set_title('Example cell: %s' % example_cell,fontsize=5)
#     # ax.set_ylabel('High',labelpad=0)
#     # ax.set_xlabel('Low',labelpad=0)
#     ax_nticks(ax,3)
#     plt.tight_layout()
#     sns.despine(fig=fig, top=True, right=True, offset=2,trim=False)
#     # my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell)
#     # my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly','BaselineCorrected'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])
