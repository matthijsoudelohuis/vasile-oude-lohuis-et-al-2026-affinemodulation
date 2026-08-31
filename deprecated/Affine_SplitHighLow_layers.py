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
# nboots                  = 0
nboots                  = 250
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)
sig_params_regress      = np.full((nCells,narealabelpairs,2),np.nan)

ndprimeboots            = 1000
# ndprimeboots = 0
dprimedata             = np.full((narealabelpairs,nCells),np.nan)
dprimesig              = np.full((narealabelpairs,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(sessions[ises].celldata[celldatafields[0]] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata[celldatafields[1]] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata[celldatafields[2]] == alp.split('-')[2])[0]

        if len(idx_N1) < params['minnneurons'] or len(idx_N2) < params['minnneurons']:
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

#%% Store results in celldata
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)
for ises in range(nSessions):
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    sessions[ises].celldata['dprime_PML23']   = dprimedata[0,idx_ses]
    sessions[ises].celldata['dprime_PML5']   = dprimedata[1,idx_ses]
    sessions[ises].celldata['sigmod_FBPML23']       = dprimesig[0,idx_ses]
    sessions[ises].celldata['sigmod_FBPML5']       = dprimesig[1,idx_ses]
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%% Feedback modulated cells:
ises = 2
iplane = 3
cellfield = 'dprime_PML23'
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_dprime_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')

cellfield = 'dprime_PML5'
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_dprime_V1L5_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')

#%%
ises = 7
iplane =4
cellfield = 'dprime_PML23'
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_dprime_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')

cellfield = 'dprime_PML5'
fig = plot_mod_plane(sessions[ises].celldata,iplane=iplane,cellfield=cellfield,
                     id_sig=False,id_looped=False,radiuslooped=False,radius=50) 
filename = 'Example_plane_dprime_V1L5_session%d_plane%d.pdf' % (ises,iplane)
# fig.savefig(os.path.join(savedir,filename),format = 'pdf',dpi=600,bbox_inches='tight')


#%% Are neurons similarly modulated by FB from L2/3 or L5?
fig,axes = plt.subplots(1,1,figsize=(4*cm,3.7*cm))
# ax = axes[ialp]
ax = axes
# ax.scatter(dprimedata[0,:],dprimedata[1,:],color='#666666',marker='.',s=3)
sns.regplot(x=dprimedata[0,:],y=dprimedata[1,:],
            scatter_kws={'s':2, 'alpha':0.5,'marker':'.','edgecolor':'none','color':'#666666'},
            line_kws={'color':"#313DCA",'lw':1,'alpha':1},ci=95,ax=ax)
ax_nticks(ax,4)
ax.set_xlabel('DPrime PM-L23')
ax.set_ylabel('DPrime PM-L5')
ax.set_xticks(np.arange(-1,1.1,0.25))
ax.set_yticks(np.arange(-1,1.1,0.25))
ax.set_ylim([-.25,0.6])
ax.set_xlim([-.25,0.6])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'dprime_%s_DiffLayers_%dsessions' % (direction,nSessions))

#%% Show pie chart of significant correlation for feedforward and feedback:
fracmat = np.empty((3,narealabelpairs))
nsigmat = np.empty((3,narealabelpairs))
ntotalmat = np.empty((3,narealabelpairs))

signorder = [-1,0,1]
signlabels = ['Neg','None','Pos']
signorder = [-1,1,0]
signlabels = ['Neg','Pos','None']   
clrs_signs = ['#0033B3','#E66804','#808080']
for ialp,alp in enumerate(arealabelpairs):
    for isign,sign in enumerate(signorder):
        idx_N = ~np.isnan(dprimedata[ialp,:])
        nsigmat[isign,ialp] = np.sum(dprimesig[ialp,idx_N]==sign)
        ntotalmat[isign,ialp] = np.sum(idx_N)
        fracmat[isign,ialp] = np.sum(dprimesig[ialp,idx_N]==sign) / np.sum(idx_N)

#Make the figure:
fig,axes = plt.subplots(1,2,figsize=(7*cm,4*cm))
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    ax.pie([fracmat[0,ialp],fracmat[1,ialp],fracmat[2,ialp]],labels=signlabels,colors=clrs_signs,autopct='%1.1f%%',
            startangle=90,counterclock=False,wedgeprops = {'linewidth': 0.8, 'edgecolor': 'black', 'alpha': 0.7},
            textprops={'fontsize': 6})
    ax.set_title('%s\nn=%d' % (legendlabels[ialp],np.sum(~np.isnan(dprimedata[ialp,:]))))
# my_savefig(fig,savedir,'Dprime_%s_Sign_Layers_PieCharts_%dsessions' % (direction,nSessions))

for ipair,pair in enumerate(np.array([[0,1]])):
    for isign,sign in enumerate(signorder[:2]):
        data = np.array([[nsigmat[isign,pair[0]], ntotalmat[isign,pair[0]]-nsigmat[isign,pair[0]]],
                         [nsigmat[isign,pair[1]], ntotalmat[isign,pair[1]]-nsigmat[isign,pair[1]]]])
        if np.any(data[:,0]):
            chival,pval = stats.chi2_contingency(data)[:2]  # p-value
            print('%s vs %s: %s: chi=%1.2f, p=%2.2g' % (legendlabels[pair[0]],legendlabels[pair[1]],
                                                      signlabels[isign],chival,pval))


#%% Make it a bar chart: 
fig,axes = plt.subplots(1,2,figsize=(2*1.6*cm,3.5*cm),sharey=True)
for isign,sign in enumerate(signorder[:2]):
    ax = axes[isign]
    ax.bar([0,1],fracmat[isign,:],width=0.6,color=clrs_signs[isign],edgecolor='black',linewidth=0.8,alpha=0.7)
    if isign == 0 and ipair == 0:
        ax.set_ylabel('Fraction of cells')
    print('%s\nn=%d cells (%s)\nn=%d cells (%s)' % (
                                                    signlabels[isign],
                                                    np.sum(~np.isnan(dprimesig[0,:])),
                                                    legendlabels[0],
                                                    np.sum(~np.isnan(dprimesig[1,:])),
                                                    legendlabels[1],
                                                    ))
    ax.set_title('%s' % signlabels[isign],fontsize=6)
    
    data = np.array([[nsigmat[isign], ntotalmat[isign]-nsigmat[isign]],
                        [nsigmat[isign], ntotalmat[isign]-nsigmat[isign]]])
    if np.any(data[:,0]):
        chival,pval = stats.chi2_contingency(data)[:2]  # p-value
        print('%s, %s vs %s: chi=%1.2f, p=%1.3e' % (signlabels[isign],legendlabels[0],legendlabels[1],
                                                    chival,pval))    
    add_stat_annotation(ax,x1=0,x2=1,y=0.18,h=0.01,p=pval)

sns.despine(fig=fig, top=True, right=True,offset=2)
for ax in axes:
    ax.set_xticks([0,1],labels=[legendlabels[0],legendlabels[1]],fontsize=4,rotation=90)

# my_savefig(fig,savedir,'Dprime_Sign_BarChart_FB_Layers_%dsessions' % nSessions)
my_savefig(fig,savedir,'Dprime_Sign_BarChart_FF_Layers_%dsessions' % nSessions)

#%% Show pie chart of significant correlation for feedforward and feedback:
fracmat = np.empty((3,narealabelpairs,nSessions))
nsigmat = np.empty((3,narealabelpairs,nSessions))
ntotalmat = np.empty((3,narealabelpairs,nSessions))

signorder = [-1,0,1]
signlabels = ['Neg','None','Pos']
signorder = [-1,1,0]
signlabels = ['Neg','Pos','None']   
clrs_signs = ['#0033B3','#E66804','#808080']
for ialp,alp in enumerate(arealabelpairs):
    for isign,sign in enumerate(signorder):
        for ises in range(nSessions):
            idx_N = np.all((~np.isnan(dprimedata[ialp,:]),
                            celldata['session_id']==sessions[ises].session_id,
                            rangeresp>params['minrangeresp']),axis=0)

            nsigmat[isign,ialp,ises] = np.sum(dprimesig[ialp,idx_N]==sign)
            ntotalmat[isign,ialp,ises] = np.sum(idx_N)
            fracmat[isign,ialp,ises] = np.sum(dprimesig[ialp,idx_N]==sign) / np.sum(idx_N)

#%% Make it a bar chart, but per session:
from scipy.stats import ttest_ind,wilcoxon,ranksums
fig,axes = plt.subplots(1,2,figsize=(2*1.6*cm,3.5*cm),sharey=True)
for isign,sign in enumerate(signorder[:2]): #signorder[:2]
# for isign,sign in [enumerate(signorder[1:])]: #signorder[:2]
    ax = axes[isign]
    ax.bar([0,1],np.nanmean(fracmat[isign],axis=1),width=0.6,color=clrs_signs[isign],edgecolor='black',linewidth=0.8,alpha=0.7)
    # ax.bar([0,1],fracmat[isign,:],width=0.6,color=clrs_signs[isign],edgecolor='black',linewidth=0.8,alpha=0.7)
    # ax.errorbar([0,1],np.nanmean(fracmat[isign],axis=1),yerr=np.nanstd(fracmat[isign],axis=1) / np.sqrt(np.sum(~np.isnan(fracmat[isign,0]))))
                # color=clrs_signs[isign],fmt='o',linewidth=0.8,alpha=0.7)
    ax.scatter(np.zeros(nSessions),fracmat[isign,0],s=3,marker='.',color='k',
               edgecolor='black',alpha=1)
    ax.scatter(np.ones(nSessions),fracmat[isign,1],s=3,marker='.',color='k',
               edgecolor='black',alpha=1)
    if isign == 0 and ipair == 0:
        ax.set_ylabel('Fraction of cells')
    print('%s\nn=%d cells (%s)\nn=%d cells (%s)' % (
                                                    signlabels[isign],
                                                    np.sum(~np.isnan(dprimesig[0,:])),
                                                    legendlabels[0],
                                                    np.sum(~np.isnan(dprimesig[1,:])),
                                                    legendlabels[1],
                                                    ))
    ax.set_title('%s' % signlabels[isign],fontsize=6)
    t,pval = ranksums(fracmat[isign,0,:],fracmat[isign,1,:],nan_policy='omit')
    print('t=%1.2f,p=%2.2g' % (t,pval))
    # add_paired_ttest_results(ax,fracmat[isign,0,:],fracmat[isign,1,:],pos=[0.5,0.8],fontsize=6)
    add_stat_annotation(ax,x1=0,x2=1,y=0.18,h=0.01,p=pval)

sns.despine(fig=fig, top=True, right=True,offset=2)
for ax in axes:
    ax.set_xticks([0,1],labels=[legendlabels[0],legendlabels[1]],fontsize=4,rotation=90)

my_savefig(fig,savedir,'Dprime_Sign_BarChart_FB_Layers_%dsessions_stats' % nSessions)
# my_savefig(fig,savedir,'Dprime_Sign_BarChart_FF_Layers_%dsessions' % nSessions)


#%% Fraction of significant multiplicative and additively modulated cells:
modsign = -1
orderversion = 1
modsign = 1
orderversion = 0
signlabel = 'excited' if modsign==1 else 'inhibited'

if orderversion==1:
    affinelabels = ['mult.','div.','add.','sub.']
else:
    affinelabels = ['mult.','add.','div.','sub.']

#%% Make the figure:
fig,axes = plt.subplots(1,1,figsize=(4.5*cm,3.8*cm))
ax = axes
sigmat = np.empty((3,2,nSessions))
countmat = np.empty((3,2,nSessions))
barwidth = 0.5
ncomparisons = 8
for iafftype in [0,1]: # 0 = multiplicative, 1 = additive
    for iaffsign,affsign in enumerate([1,-1]): # 1 = positive, -1 = negative
        fracs = np.empty((2,nSessions))
        for ialp,alp in enumerate(arealabelpairs):
            for ises in range(nSessions):
                Nsig = np.sum(np.all((
                            celldata['session_id']==sessions[ises].session_id,
                            dprimesig[ialp,:]==modsign,
                            sig_params_regress[:,ialp,iafftype]==affsign,
                            rangeresp>params['minrangeresp'],
                                ),axis=0))
                Ntotal = np.sum(np.all((
                            celldata['session_id']==sessions[ises].session_id,
                            ~np.isnan(sig_params_regress[:,ialp,iafftype]),
                            dprimesig[ialp,:]==modsign,
                            rangeresp>params['minrangeresp'],
                                ),axis=0))
                sigmat[iafftype,ialp,ises] = Nsig
                countmat[iafftype,ialp,ises] = Ntotal
                fracs[ialp,ises] = Nsig/Ntotal

            if orderversion==1: 
                xpos = iaffsign*2 + iafftype*4 + (ialp*2-1) * barwidth/1.5
            else: 
                xpos = iafftype*2 + iaffsign*4 + (ialp*2-1) * barwidth/1.5
            # print(xpos)
                # ax.bar(xpos,fracs[ialp],width=barwidth,color=clrs_arealabelpairs[ialp],edgecolor='k',linewidth=0.5)

            ax.bar(xpos,np.nanmean(fracs[ialp]),width=barwidth,color=clrs_arealabelpairs[ialp],edgecolor='black',linewidth=0.5)
            ax.scatter(np.ones(nSessions)*xpos,fracs[ialp],s=3,marker='.',color='k',
                    edgecolor='black',alpha=1)
            # ax.scatter(np.ones(nSessions),fracmat[isign,1],s=3,marker='.',color='k',
            #         edgecolor='black',alpha=1)
            # if isign == 0 and ipair == 0:
            #     ax.set_ylabel('Fraction of cells')
        if iafftype==0 and iaffsign==0: 
            print('n=%d cells from n=%d sessions (%s)\nn=%d cells from n=%d sessions (%s)' % (
                                                        np.sum(countmat[iafftype,0]),
                                                        np.sum(~np.isnan(fracs[0,:])),
                                                        legendlabels[0],
                                                        np.sum(countmat[iafftype,1]),
                                                        np.sum(~np.isnan(fracs[1,:])),
                                                        legendlabels[1],
                                                        ))
            
        if np.any(fracs>0):
            t,pval = ranksums(fracs[0,:],fracs[1,:],nan_policy='omit')
            print('t=%1.2f,p=%2.2g' % (t,pval))
            if orderversion==1: 
                xpos = iaffsign*2 + iafftype*4 + (np.array([0,1])*2-1) * barwidth/1.5
            else: 
                xpos = iafftype*2 + iaffsign*4 + (np.array([0,1])*2-1) * barwidth/1.5
            add_stat_annotation(ax,xpos[0],xpos[1],np.nanmax(fracs)+0.05,pval,h=0,fontsize=7)
ax.text(0.4, 0.9, '%s (n=%d)' % (legendlabels[0],np.sum(countmat[0,0])), fontsize=5,color=clrs_arealabelpairs[0],transform=ax.transAxes)
ax.text(0.4, 0.8, '%s (n=%d)' % (legendlabels[1],np.sum(countmat[0,1])), fontsize=5,color=clrs_arealabelpairs[1],transform=ax.transAxes)

ax_nticks(ax,4)
ax.set_xticks(np.arange(4)*2,affinelabels)
ax.set_ylabel('Fraction of %s neurons' % signlabel)
ax.set_yticks(np.arange(0,1.1,0.2))
ax.set_ylim([0,1])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=1,trim=False)
my_savefig(fig,savedir,'%s_Affinemodulation_Layers_%s_%dsessions' % (direction,signlabel,nSessions))

#%% Make the figure:
# fig,axes = plt.subplots(1,1,figsize=(4.5*cm,3.8*cm))
# ax = axes
# sigmat = np.empty((3,2))
# countmat = np.empty((3,2))
# barwidth = 0.5
# ncomparisons = 8
# for iafftype in [0,1]: # 0 = multiplicative, 1 = additive
#     for iaffsign,affsign in enumerate([1,-1]): # 1 = positive, -1 = negative
#         fracs = np.empty((2))
#         for ialp,alp in enumerate(arealabelpairs):
#             Nsig = np.sum(np.all((
#                         dprimesig[ialp,:]==modsign,
#                         sig_params_regress[:,ialp,iafftype]==affsign,
#                         rangeresp>params['minrangeresp'],
#                             ),axis=0))
#             Ntotal = np.sum(np.all((
#                         ~np.isnan(sig_params_regress[:,ialp,iafftype]),
#                         dprimesig[ialp,:]==modsign,
#                         rangeresp>params['minrangeresp'],
#                             ),axis=0))
#             sigmat[iafftype,ialp] = Nsig
#             countmat[iafftype,ialp] = Ntotal
#             fracs[ialp] = Nsig/Ntotal

#             if orderversion==1: 
#                 xpos = iaffsign*2 + iafftype*4 + (ialp*2-1) * barwidth/1.5
#             else: 
#                 xpos = iafftype*2 + iaffsign*4 + (ialp*2-1) * barwidth/1.5
#             print(xpos)
#             ax.bar(xpos,fracs[ialp],width=barwidth,color=clrs_arealabelpairs[ialp],edgecolor='k',linewidth=0.5)
        
#         if np.any(fracs>0):
#             pval = stats.chi2_contingency([[sigmat[iafftype,0], countmat[iafftype,0]-sigmat[iafftype,0]],
#                                 [sigmat[iafftype,1], countmat[iafftype,1]-sigmat[iafftype,1]]])[1]
#             pval = np.clip(pval*ncomparisons,0,1)
#             # print(pval)
#             if orderversion==1: 
#                 xpos = iaffsign*2 + iafftype*4 + (np.array([0,1])*2-1) * barwidth/1.5
#             else: 
#                 xpos = iafftype*2 + iaffsign*4 + (np.array([0,1])*2-1) * barwidth/1.5
#             add_stat_annotation(ax,xpos[0],xpos[1],np.max(fracs)+0.05,pval,h=0,fontsize=7)
# ax.text(0.4, 0.9, '%s (n=%d)' % (legendlabels[0],countmat[0,0]), fontsize=5,color=clrs_arealabelpairs[0],transform=ax.transAxes)
# ax.text(0.4, 0.8, '%s (n=%d)' % (legendlabels[1],countmat[0,1]), fontsize=5,color=clrs_arealabelpairs[1],transform=ax.transAxes)

# ax_nticks(ax,4)
# ax.set_xticks(np.arange(4)*2,affinelabels)
# ax.set_ylabel('Fraction of %s neurons' % signlabel)
# ax.set_yticks(np.arange(0,1.1,0.2))
# ax.set_ylim([0,1])
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=1,trim=False)
# # my_savefig(fig,savedir,'%s_Affinemodulation_Layers_%s_cells_%dsessions' % (direction,signlabel,nSessions))


#%% Fraction of significant multiplicative and additively modulated cells:
# sign = 1
# fig,axes = plt.subplots(1,1,figsize=(6*cm,4*cm))
# ax = axes
# idx_N =  rangeresp>params['minrangeresp']
# sigmat = np.empty((3,2))
# countmat = np.empty((3,2))
# for itype,(mult,add) in enumerate(zip([1,0,1],[0,1,1])):
#     for ialp,alp in enumerate(arealabelpairs):
#         Nsig = np.sum(np.all((
#                     sig_params_regress[idx_N,ialp,0]==mult,
#                     sig_params_regress[idx_N,ialp,1]==add,
#                     dprimesig[ialp,idx_N]==sign,
#                         ),axis=0))
#         Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
#         sigmat[itype,ialp] = Nsig
#         countmat[itype,ialp] = Ntotal
#         frac = Nsig/Ntotal
#         xpos = itype*2 + ialp
#         ax.bar(xpos,frac,width=0.8,color=clrs_arealabelpairs[ialp])
#     pval = stats.chi2_contingency([[sigmat[itype,0], countmat[itype,0]-sigmat[itype,0]],
#                             [sigmat[itype,1], countmat[itype,1]-sigmat[itype,1]]])[1]
#     add_stat_annotation(ax,xpos-1,xpos+0,frac+0.01,pval,h=0,fontsize=9)
# ax_nticks(ax,4)
# ax.text(0.6, 0.9, '%s (n=%d)' % (legendlabels[0],np.sum(dprimesig[0,idx_N]==sign)), color=clrs_arealabelpairs[0],transform=ax.transAxes)
# ax.text(0.6, 0.8, '%s (n=%d)' % (legendlabels[1],np.sum(dprimesig[1,idx_N]==sign)), color=clrs_arealabelpairs[1],transform=ax.transAxes)
# ax.set_xticks(np.arange(3)*2+0.5,['Mult','Add','Both'])
# ax.set_ylabel('Fraction of + correlated cells')
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'%s_affinemodulation_sig_posmod_Layers_%dsessions' % (direction,nSessions))

# #%% Fraction of significant multiplicative and additively modulated cells:
# sign = -1
# fig,axes = plt.subplots(1,1,figsize=(6*cm,4*cm))
# ax = axes
# idx_N =  rangeresp>params['minrangeresp']
# sigmat = np.empty((3,2))
# countmat = np.empty((3,2))
# for itype,(mult,add) in enumerate(zip([-1,0,-1],[0,-1,-1])):
#     for ialp,alp in enumerate(arealabelpairs):
#         Nsig = np.sum(np.all((
#                     sig_params_regress[idx_N,ialp,0]==mult,
#                     sig_params_regress[idx_N,ialp,1]==add,
#                     dprimesig[ialp,idx_N]==sign,
#                         ),axis=0))
#         Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
#         sigmat[itype,ialp] = Nsig
#         countmat[itype,ialp] = Ntotal
#         frac = Nsig/Ntotal
#         xpos = itype*2 + ialp
#         ax.bar(xpos,frac,width=0.8,color=clrs_arealabelpairs[ialp])
#     pval = stats.chi2_contingency([[sigmat[itype,0], countmat[itype,0]-sigmat[itype,0]],
#                             [sigmat[itype,1], countmat[itype,1]-sigmat[itype,1]]])[1]
#     add_stat_annotation(ax,xpos-1,xpos+0,frac+0.01,pval,h=0,fontsize=9)
#     ax_nticks(ax,3)
    
# ax.text(0.6, 0.9, '%s (n=%d)' % (legendlabels[0],np.sum(dprimesig[0,idx_N]==sign)), color=clrs_arealabelpairs[0],transform=ax.transAxes)
# ax.text(0.6, 0.8, '%s (n=%d)' % (legendlabels[1],np.sum(dprimesig[1,idx_N]==sign)), color=clrs_arealabelpairs[1],transform=ax.transAxes)

# ax.set_xticks(np.arange(3)*2+0.5,['Div','Sub','Both'])
# ax.set_ylabel('Fraction of - correlated cells')
# plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'%s_affinemodulation_sig_negmod_Layers_%dsessions' % (direction,nSessions))

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
                dprimesig[ialp,:]==1,
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
    
    # h,p = stats.ttest_ind(params_regress[idx_N,0,iparam],
    #                         params_regress[idx_N,1,iparam],nan_policy='omit')
    h,p = stats.mannwhitneyu(params_regress[idx_N,0,iparam],
                            params_regress[idx_N,1,iparam],alternative='two-sided',nan_policy='omit')
    p = np.clip(p * narealabelpairs * 2,0,1) #bonferroni + clip
    print(p)
    # ax.text(0.6, 0.15, '%s,p=%1.2f' % (get_sig_asterisks(p,return_ns=True),p), transform=ax.transAxes,fontsize=9)
    ax.text(0.45, 0.5, '%s' % (get_sig_asterisks(p,return_ns=True)),
            transform=ax.transAxes)
    ax.set_yticks([0,0.5,1.0])
    ax.set_xlim(xlims)
    ax.set_ylabel('Cumulative fraction of cells')

    # ax.legend(handles,legendlabels,frameon=False,loc='center right')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=2)

# my_savefig(fig,savedir,'%s_affinemodulation_layers_posmod_cumhistcoefs_%dGRsessions' % (direction,nSessions))

#%% Cumulative
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
                    dprimesig[ialp,:]==modsign,

                    # np.any(dprimesig==modsign,axis=0),
                    ),axis=0)
            datatoplot = np.clip(params_regress[idx_N,ialp,iparam],bins[1],bins[-2])
            sns.histplot(data=datatoplot,element='step',
                        color=clrs_arealabelpairs[ialp],
                        alpha=1,linewidth=1.5,ax=ax,stat='probability',bins=bins,cumulative=True,fill=False)
            handles.append(ax.plot(np.nanmean(datatoplot),0.95,markersize=6,
                    color=clrs_arealabelpairs[ialp],marker='v')[0])
            ncells = np.sum(~np.isnan(datatoplot))

            ax.text(0.4, 0.1+ialp*0.1, '%s (n=%d)' % (legendlabels[ialp],ncells), 
                    transform=ax.transAxes,fontsize=5,color=clrs_arealabelpairs[ialp])
            
        idx_N = np.all((
                rangeresp>params['minrangeresp'],
                ),axis=0)
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

    my_savefig(fig,savedir,'%s_Layers_affinemodulation_dprime_%s_cumhistcoefs_%dGRsessions' % (direction,signlabel,nSessions))


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
                # dprimesig[ialp,:]==1,
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
                # dprimesig[ialp,:]==-1,
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
