#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress,ranksums
from scipy import stats

os.chdir('e:\\Python\\oudelohuis-et-al-2026-anatomicalsubspace')

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
sessions,nSessions   = filter_sessions(protocols = 'GR',filter_noiselevel=False)
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
fig,ax = plt.subplots(1,1,figsize=(3,3),sharex=True,sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    # ax.scatter(radii, np.nanmean(fracnonlooped[ialp,:,:],axis=1), marker='o', color=clrs_arealabelpairs[ialp])
    # ax.plot(radii,np.nanmean(error_cv[:,ialp,:,:],axis=(2)),color=clrs_arealabelpairs[ialp], linewidth=0.5)
    shaded_error(radii,fracnonlooped[ialp,:,:].T,center='mean',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)
    # shaded_error(radii,fracnonlooped[ialp,:,:].T,center='median',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)
    # ax.plot(radii,fracnonlooped[ialp,:,:],color=clrs_arealabelpairs[ialp])
    ax.set_ylim([0,1])
    # ax.set_xlim([0,popsizes[-1]+1])
    # ax.set_xticks(radii)
    ax.set_xticks(np.arange(0,400,50))
    ax.set_xlabel('Min dist to looped cell (um)')
    if ialp == 0: 
        ax.set_ylabel('Fraction nonlooped cells')
    # ax.set_title(legendlabels[ialp],fontsize=12)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Fraction_nonlooped_distance_V1PM_%dsessions' % nSessions)

#%% Compute the fraction of looped neurons with at least one neighbour within a radius:
arealabelpairs      = ['PMlabL2/3','V1labL2/3']
narealabelpairs     = len(arealabelpairs)
legendlabels        = ['PM','V1']

radii               = np.arange(0,100,5)
fraclooped          = np.zeros((narealabelpairs,len(radii),nSessions))

for ises in range(nSessions):
    for iradius,radius in enumerate(radii):
        for ialp,alp in enumerate(arealabelpairs):
            idx_N = sessions[ises].celldata['arealayerlabel'] == alp
            idx_L = sessions[ises].celldata['redcell']==0
            idx_nearby = np.any(sessions[ises].distmat_xyz[np.ix_(idx_N,idx_L)]<radius,axis=1)
            fraclooped[ialp,iradius,ises] = np.sum(idx_nearby) / np.sum(idx_N)


#%% Show the fraction of nonlooped neurons within a radius from looped cells:
fig,ax = plt.subplots(1,1,figsize=(3,3),sharex=True,sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    # ax.scatter(radii, np.nanmean(fracnonlooped[ialp,:,:],axis=1), marker='o', color=clrs_arealabelpairs[ialp])
    # ax.plot(radii,np.nanmean(error_cv[:,ialp,:,:],axis=(2)),color=clrs_arealabelpairs[ialp], linewidth=0.5)
    shaded_error(radii,fraclooped[ialp,:,:].T,center='mean',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)
    # shaded_error(radii,fracnonlooped[ialp,:,:].T,center='median',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)
    # ax.plot(radii,fracnonlooped[ialp,:,:],color=clrs_arealabelpairs[ialp])
    ax.set_ylim([0,1])
    # ax.set_xlim([0,popsizes[-1]+1])
    # ax.set_xticks(radii)
    ax.set_xticks(radii[::2])
    ax.set_xlabel('Min dist to looped cell (um)')
    if ialp == 0: 
        ax.set_ylabel('Fraction looped cells included')
    # ax.set_title(legendlabels[ialp],fontsize=12)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Fraction_nonlooped_distance_V1PM_%dsessions' % nSessions)


#%%
# without subsampling to equalize numbers of neruons
# p<0.001
# nearby neurons only, <50 um
# maxnoiselevel=20
# minneurons=10

# #%%  #assign arealayerlabel
# for ises in range(nSessions):   
#     # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
#     sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
#     sessions[ises].celldata['arealayerlabel']   = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 
#     sessions[ises].celldata['arealayer']        = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 

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

#criteria for selecting still trials:
# maxvideome              = 0.2
# maxrunspeed             = 5
# alphathr                = 0.001 #threshold for correlation with cross area rate
params['alpha_crossrate'] = 0.0001
params['activitymetric'] = 'ratio' #'mean','ratio','difference'
params['activitymetric'] = 'difference' #'mean','ratio','difference'
minnneurons             = 10
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
        
        idx_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                    # sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]
        idx_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                    # sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                    sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]

        # idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        # idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
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
        # tempcorr          = np.array([ranksums(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        # tempsig          = np.array([ranksums(meanpopact,respdata[n,:])[1] for n in idx_N3])
        
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<params['alpha_crossrate']) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

#%% Show some example neurons:

# #%% use 
# ialp = 0
# clrs_arealabelpairs = ['green','purple']
# legendlabels        = ['FF','FB']

# #%% Get good multiplicatively modulated cells by FF or FB:
# #mutliplicative: 
# idx_examples = np.all((params_regress[:,ialp,0]>np.nanpercentile(params_regress[:,ialp,0],80),
#                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                        ),axis=0)
# #divisive:
# idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],50),
#                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],50),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                        ),axis=0)

# print(celldata['cell_id'][idx_examples])

# example_cell      = np.random.choice(celldata['cell_id'][idx_examples])
# example_cells     = np.random.choice(celldata['cell_id'][idx_examples],2)

# #%% Get good additively modulated cells by FB: 
# #additive:
# idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],80),
#                        params_regress[:,ialp,1]>np.nanpercentile(params_regress[:,ialp,1],70),
#                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
#                        ),axis=0)
# # #subtractive:
# # idx_examples = np.all((params_regress[:,ialp,0]<np.nanpercentile(params_regress[:,ialp,0],70),
# #                        params_regress[:,ialp,1]<np.nanpercentile(params_regress[:,ialp,1],25),
# #                        params_regress[:,ialp,2]>np.nanpercentile(params_regress[:,ialp,2],80),
# #                        ),axis=0)
# print(celldata['cell_id'][idx_examples])

# example_cell      = np.random.choice(celldata['cell_id'][idx_examples])

#%% 

######  ####### ####### #######    ######     #    ######  ### #     #  #####  
#     # #     # #     #    #       #     #   # #   #     #  #  #     # #     # 
#     # #     # #     #    #       #     #  #   #  #     #  #  #     # #       
######  #     # #     #    #       ######  #     # #     #  #  #     #  #####  
#     # #     # #     #    #       #   #   ####### #     #  #  #     #       # 
#     # #     # #     #    #       #    #  #     # #     #  #  #     # #     # 
######  ####### #######    #       #     # #     # ######  ###  #####   #####  


#%% 
# for ises in range(nSessions):
#     sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'],metric='xyz')
#     # sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'],metric='xy')
# celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

# #%%  #assign arealayerlabel
for ises in range(nSessions):   
    sessions[ises].celldata = sessions[ises].assign_layer2(splitdepth=250)
    # sessions[ises].celldata = sessions[ises].assign_layer()
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 
    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%%
# from utils.tuning import comp_grating_responsive
# sessions = comp_grating_responsive(sessions,pthr = 0.001)
# celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%% Bootstrapped comparison of correlations and significant correlations with other area: 
# The distribution of correlations is compared to the loop correlation distribution.
# The fraction of significantly positive and negative as well. 
radii           = np.arange(30,200,20)
nradii          = len(radii)

idx_PMlab       = np.where(celldata['arealayerlabel'] == 'PMlabL2/3')[0]
idx_V1lab       = np.where(celldata['arealayerlabel'] == 'V1labL2/3')[0]

idx_PMlab       = np.where(np.all((
                                    celldata['arealayerlabel'] == 'PMlabL2/3',
                                    rangeresp>params['minrangeresp'],
                                    # celldata['noise_level']<params['maxnoiselevel'],
                                    ),axis=0))[0]
idx_V1lab       = np.where(np.all((
                                    celldata['arealayerlabel'] == 'V1labL2/3',
                                    rangeresp>params['minrangeresp'],
                                    # celldata['noise_level']<params['maxnoiselevel'],
                                    ),axis=0))[0]
nPMlab          = len(idx_PMlab)
nV1lab          = len(idx_V1lab)
idx_PMlab_allnear  = np.full((nPMlab,2,200),np.nan) #store up to 100 nearby nonlabeled cells for each labeled cell, get idx and distance
idx_V1lab_allnear  = np.full((nV1lab,2,200),np.nan)

for iN,N in tqdm(enumerate(idx_PMlab),total=nPMlab,desc='Finding PM neurons nearby labeled cells'):
    #get index of which session this labeled cell comes from:
    ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    #get index of labeled cell in this session
    idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
    #get index of all unlabeled cells in this session that are nearby this particular labeled cell
    idx_nearby_ses = np.where(np.all((
                                        np.squeeze(sessions[ises].distmat_xyz[idx_N_ses,:]<np.max(radii)),
                                        # np.squeeze(sessions[ises].distmat_xy[idx_N_ses,:]<np.max(radii)),
                                        rangeresp[idx_ses]>params['minrangeresp'],
                                        # sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        # sessions[ises].celldata['redcell']==0,
                                        # sessions[ises].celldata['arealayerlabel'] == 'PMunlL2/3',
                                        # sessions[ises].celldata['vis_resp'],
                                        ),axis=0))[0]
    
    # tempdist = sessions[ises].distmat_xyz[np.ix_(idx_N_ses,idx_nearby_ses)].squeeze()
    # sortidx = np.argsort(tempdist)
    # print(len(idx_nearby_ses))
    # tempdist = tempdist[sortidx]
    # idx_nearby_ses = idx_nearby_ses[sortidx] #sort by distance
    idx_nearby_ses = idx_nearby_ses[:200] #limit to first 100 nearby cells
    idx_nearby_all = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
    idx_PMlab_allnear[iN,0,:len(idx_nearby_ses)] = idx_nearby_all
    idx_PMlab_allnear[iN,1,:len(idx_nearby_ses)] = sessions[ises].distmat_xyz[np.ix_(idx_N_ses,idx_nearby_ses)]
    # idx_PMlab_allnear[iN,1,:len(idx_nearby_ses)] = sessions[ises].distmat_xy[np.ix_(idx_N_ses,idx_nearby_ses)]

for iN,N in tqdm(enumerate(idx_V1lab),total=nV1lab,desc='Finding V1 neurons nearby labeled cells'):
    #get index of which session this labeled cell comes from:
    ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
    #get index of all cells in this session
    idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
    #get index of labeled cell in this session
    idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
    #get index of all unlabeled cells in this session that are nearby this particular labeled cell
    idx_nearby_ses = np.where(np.all((
                                        np.squeeze(sessions[ises].distmat_xyz[idx_N_ses,:]<np.max(radii)),
                                        # np.squeeze(sessions[ises].distmat_xy[idx_N_ses,:]<np.max(radii)),
                                        rangeresp[idx_ses]>params['minrangeresp'],
                                        # sessions[ises].celldata['redcell']==0,
                                        # sessions[ises].celldata['arealayerlabel'] == 'V1unlL2/3',
                                        # sessions[ises].celldata['vis_resp'],
                                        # sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        ),axis=0))[0]
    # tempdist = sessions[ises].distmat_xyz[np.ix_(idx_N_ses,idx_nearby_ses)].squeeze()
    # sortidx = np.argsort(tempdist)
    # tempdist = tempdist[sortidx]
    # idx_nearby_ses = idx_nearby_ses[sortidx] #sort by distance
    idx_nearby_ses = idx_nearby_ses[:200] #limit to first 100 nearby cells
    idx_nearby_all = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
    idx_V1lab_allnear[iN,0,:len(idx_nearby_ses)] = idx_nearby_all
    idx_V1lab_allnear[iN,1,:len(idx_nearby_ses)] = sessions[ises].distmat_xyz[np.ix_(idx_N_ses,idx_nearby_ses)]
    # idx_V1lab_allnear[iN,1,:len(idx_nearby_ses)] = sessions[ises].distmat_xy[np.ix_(idx_N_ses,idx_nearby_ses)]

#%% 
nboots          = 100
nboots          = 1000

loopfrac        = np.full((2,3,nradii),np.nan) # FF vs FB, +corr vs -corr vs modulated
loopmean        = np.full((2,nradii),np.nan) # FF vs FB, +corr vs -corr
loopmean_abs    = np.full((2,nradii),np.nan) # FF vs FB, +corr vs -corr

bootfrac        = np.full((2,3,nradii,nboots),np.nan) # FF vs FB, +corr vs -corr
bootmean        = np.full((2,nradii,nboots),np.nan) # FF vs FB, +corr vs -corr
bootmean_abs    = np.full((2,nradii,nboots),np.nan) # FF vs FB, +corr vs -corr

binedges        = np.linspace(-1,1,50)
nhistbins       = len(binedges)-1
loophist        = np.full((2,nhistbins,nradii),np.nan) # FF vs FB, +corr vs -corr
boothist        = np.full((2,nhistbins,nradii,nboots),np.nan) # FF vs FB, +corr vs -corr

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

    loopfrac[iapl,0,irad] = np.sum(corrsig_cells[1,idx_PMlab_haswithinradius]==1) / len(idx_PMlab_haswithinradius)
    loopfrac[iapl,1,irad] = np.sum(corrsig_cells[1,idx_PMlab_haswithinradius]==-1) / len(idx_PMlab_haswithinradius)
    loopfrac[iapl,2,irad] = (loopfrac[iapl,0,irad]+loopfrac[iapl,1,irad])
    
    loopmean[iapl,irad] = np.nanmean(corrdata_cells[1,idx_PMlab_haswithinradius])
    loopmean_abs[iapl,irad] = np.nanmean(np.abs(corrdata_cells[1,idx_PMlab_haswithinradius]))

    histcounts      = np.histogram(corrdata_cells[1,idx_PMlab_haswithinradius],bins=binedges)[0]
    loophist[iapl,:,irad]   = np.cumsum(histcounts)/np.sum(histcounts)

    for iboot in range(nboots):
        idx_boot = np.full(len(idx_PMlab),np.nan)
        for iN,N in enumerate(idx_PMlab):
            idx_within_radius = idx_PMlab_allnear[iN,0,idx_PMlab_allnear[iN,1,:] <= radius]
            if hasnearby[iN]:
                idx_boot[iN] = np.random.choice(idx_within_radius,1)
        idx_boot                    = idx_boot[hasnearby].astype(int)
        bootfrac[iapl,0,irad,iboot] = np.sum(corrsig_cells[0,idx_boot]==1) / len(idx_boot) #compute fraction of sig pos for this boot
        bootfrac[iapl,1,irad,iboot] = np.sum(corrsig_cells[0,idx_boot]==-1) / len(idx_boot)
        bootfrac[iapl,2,irad,iboot] = (bootfrac[iapl,0,irad,iboot]+bootfrac[iapl,1,irad,iboot])

        bootmean[iapl,irad,iboot] = np.nanmean(corrdata_cells[0,idx_boot])
        bootmean_abs[iapl,irad,iboot] = np.nanmean(np.abs(corrdata_cells[0,idx_boot]))

        histcounts = np.histogram(corrdata_cells[0,idx_boot],bins=binedges)[0]
        boothist[iapl,:,irad,iboot] = np.cumsum(histcounts)/np.sum(histcounts)

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

    loopfrac[iapl,0,irad] = np.sum(corrsig_cells[3,idx_V1lab_haswithinradius]==1) / len(idx_V1lab_haswithinradius)
    loopfrac[iapl,1,irad] = np.sum(corrsig_cells[3,idx_V1lab_haswithinradius]==-1) / len(idx_V1lab_haswithinradius)
    loopfrac[iapl,2,irad] = (loopfrac[iapl,0,irad]+loopfrac[iapl,1,irad])
    
    loopmean[iapl,irad] = np.nanmean(corrdata_cells[3,idx_V1lab_haswithinradius])
    loopmean_abs[iapl,irad] = np.nanmean(np.abs(corrdata_cells[3,idx_V1lab_haswithinradius]))
    histcounts      = np.histogram(corrdata_cells[3,idx_V1lab_haswithinradius],bins=binedges)[0]
    loophist[iapl,:,irad]   = np.cumsum(histcounts)/np.sum(histcounts)

    for iboot in range(nboots):
        idx_boot = np.full(len(idx_V1lab),np.nan)
        for iN,N in enumerate(idx_V1lab):
            idx_within_radius = idx_V1lab_allnear[iN,0,idx_V1lab_allnear[iN,1,:] <= radius]
            if hasnearby[iN]:
                idx_boot[iN] = np.random.choice(idx_within_radius,1)
        idx_boot                    = idx_boot[hasnearby].astype(int)
        bootfrac[iapl,0,irad,iboot] = np.sum(corrsig_cells[2,idx_boot]==1) / len(idx_boot) #compute fraction of sig pos for this boot
        bootfrac[iapl,1,irad,iboot] = np.sum(corrsig_cells[2,idx_boot]==-1) / len(idx_boot)
        bootfrac[iapl,2,irad,iboot] = (bootfrac[iapl,0,irad,iboot]+bootfrac[iapl,1,irad,iboot])

        bootmean[iapl,irad,iboot] = np.nanmean(corrdata_cells[2,idx_boot])
        bootmean_abs[iapl,irad,iboot] = np.nanmean(np.abs(corrdata_cells[2,idx_boot]))
        histcounts = np.histogram(corrdata_cells[2,idx_boot],bins=binedges)[0]
        boothist[iapl,:,irad,iboot] = np.cumsum(histcounts)/np.sum(histcounts)

#%% Plotting bootstrapped results across different radii:

legendlabels        = ['FF','FB']
axisbuffer          = 0.025
lw                  = 2

subplotlabels       = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.','Frac. Mod.'])
loopdata_subplots   = np.stack((loopmean,loopmean_abs,loopfrac[:,0],loopfrac[:,1],loopfrac[:,2]),axis=2)
bootdata_subplots   = np.stack((bootmean,bootmean_abs,bootfrac[:,0],bootfrac[:,1],bootfrac[:,2]),axis=2)
nmetrics            = len(subplotlabels)
params['ci']        = 95
fig,axes = plt.subplots(2,nmetrics,figsize=(nmetrics*2.2,4))
clrs_arealabelpairs = [ '#9933FF','#00CC99']

for ialp in range(2):
    for imetric in range(nmetrics):
        axidx = imetric
        ax  = axes[ialp,axidx]
        ax.plot(radii,loopdata_subplots[ialp,:,imetric],color=clrs_arealabelpairs[ialp],linewidth=lw,marker='.',markersize=8)
        tempdata = bootdata_subplots[np.ix_([ialp],range(nradii),[imetric],range(nboots))].squeeze()
        
        ax.fill_between(radii, np.percentile(tempdata,(100-params['ci'])/2,axis=1), np.percentile(tempdata,params['ci']+(100-params['ci'])/2,axis=1), color='grey',alpha=0.25)
        ax_nticks(ax,3)
        for irad,radius in enumerate(radii):
            ax.text((irad+1)/len(radii),0,'n=%d'%nhasnearby[ialp,irad],fontsize=6,
                    ha='center',va='bottom',transform=ax.transAxes,color='grey',rotation=45)
        if ialp == 0:
            axes[ialp,axidx].set_title(subplotlabels[imetric])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Looped_Modulations_Bootstrap_Radii')


#%% Plotting bootstrapped results at a specific radius:

radius              = 50
irad                = np.where(radii==radius)[0][0]

legendlabels        = ['FF','FB']
axisbuffer          = 0.025
lw                  = 2
nbins               = 30
subplotlabels       = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.','Frac. Mod.'])
loopdata_subplots   = np.stack((loopmean[:,irad],loopmean_abs[:,irad],loopfrac[:,0,irad],loopfrac[:,1,irad],loopfrac[:,2,irad]),axis=1)
bootdata_subplots   = np.stack((bootmean[:,irad],bootmean_abs[:,irad],bootfrac[:,0,irad],bootfrac[:,1,irad],bootfrac[:,2,irad]),axis=1)
nmetrics            = len(subplotlabels)

fig,axes = plt.subplots(2,nmetrics+1,figsize=(nmetrics*2.2,4))

for ialp in range(2):
    axes[ialp,0].plot(binedges[:-1],loophist[ialp,:,irad],color=clrs_arealabelpairs[ialp])
    tempdata = boothist[np.ix_([ialp],range(nhistbins),[irad],range(nboots))].squeeze()
    shaded_error(binedges[:-1],np.nanmean(tempdata,axis=1),np.nanstd(tempdata,axis=1),
                    ax=axes[ialp,0],color='grey')
    axes[ialp,0].set_xlim([binedges[np.where(loophist[ialp,:]>0)[0][0]],binedges[np.where(loophist[ialp,:]>0.999)[0][0]]])
    axes[ialp,0].set_ylim([0,1])
    axes[ialp,0].set_ylabel(legendlabels[ialp],fontsize=15,fontweight='bold',color=clrs_arealabelpairs[ialp])
    if ialp == 0:
        axes[ialp,0].set_title('Corr. coeff.')
    
    for imetric in range(nmetrics):
        axidx = imetric+1
        ax = axes[ialp,axidx]
        ax.axvline(loopdata_subplots[ialp,imetric],color=clrs_arealabelpairs[ialp],linewidth=lw)
        bins = np.linspace(np.percentile(bootdata_subplots[ialp,imetric],0)-axisbuffer,
                           np.percentile(bootdata_subplots[ialp,imetric],100)+axisbuffer,nbins)
        sns.histplot(bootdata_subplots[ialp,imetric,:],ax=ax,bins=bins,element='step',stat='probability',color='grey')
        xlims = [np.min([np.percentile(bootdata_subplots[ialp,imetric],0),loopdata_subplots[ialp,imetric]])-axisbuffer,
                 np.max([np.percentile(bootdata_subplots[ialp,imetric],100),loopdata_subplots[ialp,imetric]])+axisbuffer]
        ax.set_xlim(xlims) #set lim to extremes of bootstrapped data + small buffer
        pval = np.sum(bootdata_subplots[ialp,imetric,:]>loopdata_subplots[ialp,imetric])/len(bootdata_subplots[ialp,imetric,:])
        ax.text(loopdata_subplots[ialp,imetric],ax.get_ylim()[1]*0.8,get_sig_asterisks(np.min([pval,1-pval]),return_ns=True),fontsize=12,color=clrs_arealabelpairs[ialp])
        if ialp == 0:
            ax.set_title(subplotlabels[imetric])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Looped_Modulations_Bootstrap_Radius%d' % (radius))

#%% 








#%%
                                            
#       ####   ####  #####  # #    #  ####  
#      #    # #    # #    # # ##   # #    # 
#      #    # #    # #    # # # #  # #      
#      #    # #    # #####  # #  # # #  ### 
#      #    # #    # #      # #   ## #    # 
######  ####   ####  #      # #    #  ####  


#%% 
for ises in range(nSessions):
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'],metric='xyz')
celldata = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%% Set threshold for significant correlations based on correlation value: 
# corrsig_cells = np.full((narealabelpairs,nCells),np.nan)
# corrsig_cells[corrdata_cells>0.25]           = 1
# corrsig_cells[corrdata_cells<-0.25]          = -1

#%%
fracdata = np.full((narealabelpairs,2,nSessions),np.nan)

for ises in range(nSessions):
    idx_ses = np.isin(celldata['session_id'],sessions[ises].session_id)

    for ialp,alp in enumerate(arealabelpairs):
        idx_N = np.all((idx_ses,
                        ~np.isnan(corrsig_cells[ialp,:]),
                        celldata['nearby'],
                        rangeresp>params['minrangeresp'],
                        ),axis=0)
        fracdata[ialp,0,ises] = np.sum(corrsig_cells[ialp,idx_N]==1) / np.sum(idx_N)
        fracdata[ialp,1,ises] = np.sum(corrsig_cells[ialp,idx_N]==-1) / np.sum(idx_N)
        # fracdata[ialp,1,ises] = np.sum(corrsig_cells[ialp,idx_ses]==-1) / np.sum(~np.isnan(corrsig_cells[ialp,idx_ses]))

clrs = ['black','red']
axtitles = np.array(['FF: +corr','FF: -corr', 'FB: +corr','FB: -corr'])

fig,axes = plt.subplots(1,4,figsize=(12,3))
ax = axes[0]
sns.barplot(data=fracdata[:2,0,:].T,palette=clrs,ax=ax,alpha=0.3)
sns.stripplot(data=fracdata[:2,0,:].T,palette=clrs,ax=ax)
h,p = stats.ttest_rel(fracdata[0,0,:],fracdata[1,0,:],nan_policy='omit')
add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[:2,0,:]), p, h=0)
ax.set_xticklabels(arealabelpairs[:2])
print(p)

ax = axes[1]
sns.barplot(data=fracdata[:2,1,:].T,palette=clrs,ax=ax,alpha=0.3)
sns.stripplot(data=fracdata[:2,1,:].T,palette=clrs,ax=ax)
h,p = stats.ttest_rel(fracdata[0,1,:],fracdata[1,1,:],nan_policy='omit')
add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[:2,1,:]), p, h=0)
ax.set_xticklabels(arealabelpairs[:2])
print(p)

ax = axes[2]
sns.barplot(data=fracdata[2:,0,:].T,palette=clrs,ax=ax,alpha=0.3)
sns.stripplot(data=fracdata[2:,0,:].T,palette=clrs,ax=ax)
h,p = stats.ttest_rel(fracdata[2,0,:],fracdata[3,0,:],nan_policy='omit')
add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[2:,0,:]), p, h=0)
ax.set_xticklabels(arealabelpairs[2:])
print(p)

ax = axes[3]
sns.barplot(data=fracdata[2:,1,:].T,palette=clrs,ax=ax,alpha=0.3)
sns.stripplot(data=fracdata[2:,1,:].T,palette=clrs,ax=ax)
h,p = stats.ttest_rel(fracdata[2,1,:],fracdata[3,1,:],nan_policy='omit')
add_stat_annotation(ax, 0.2, .8, np.nanmean(fracdata[2:,1,:]), p, h=0)
ax.set_xticklabels(arealabelpairs[2:])
print(p)

for iax,ax in enumerate(axes):
    ax.set_title(axtitles[iax])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_affinemodulation_FracSig_%dsessions' % (nSessions), formats = ['png'])

#%%
legendlabels    = ['nonlooped','looped']

fracmat         = np.full((3,3,2,3),np.nan)
nsigmat         = np.full((3,3,2,2),np.nan)
ntotalmat       = np.full((3,3,2,2),np.nan)
testmat         = np.full((3,3,2),np.nan)
ncomparisons    = 1

for ialp,alp in enumerate(arealabelpairs):
    idir = ialp//2
    icat = ialp%2
    idx_N = np.all((
                    rangeresp>params['minrangeresp'],
                    celldata['nearby'],
                    # celldata['noise_level']<maxnoiselevel,
                    # corrsig_cells[icat,:],
                    # corrsig_cells[ialp,:] !=1,
                     ),axis=0)
    for imult, mult in enumerate([1,0,-1]):
        for iadd, add in enumerate([-1,0,1]):
            Nsig = np.sum(np.all((
                                sig_params_regress[idx_N,ialp,0]==mult,
                                sig_params_regress[idx_N,ialp,1]==add,
                                ),axis=0))
            Ntotal = np.sum(~np.isnan(sig_params_regress[idx_N,ialp,0]))
            frac = (Nsig/Ntotal) * 100

            nsigmat[imult,iadd,idir,icat] = Nsig
            ntotalmat[imult,iadd,idir,icat] = Ntotal
            fracmat[imult,iadd,idir,icat] = frac
fracmat[:,:,:,2] = fracmat[:,:,:,1] - fracmat[:,:,:,0]
# fracmat[:,:,:,2] = fracmat[:,:,:,1] - fracmat[:,:,:,0]

for i in range(2):
    for imult, mult in enumerate([1,0,-1]):
        for iadd, add in enumerate([-1,0,1]):
            data = np.array([[nsigmat[imult,iadd,i,0], ntotalmat[imult,iadd,i,0]-nsigmat[imult,iadd,i,0]],
                            [nsigmat[imult,iadd,i,1], ntotalmat[imult,iadd,i,1]-nsigmat[imult,iadd,i,1]]])
            if np.all(data[:,0]==0): 
                continue
            testmat[imult,iadd,i] = stats.chi2_contingency(data)[1]  # p-value
testmat = testmat * ncomparisons  #bonferroni correction

fig,axes = plt.subplots(2,3,figsize=(9,6))
for idir in range(2):
    for icat in range(3):
        ax = axes[idir,icat]
        if icat < 2:
            vmin,vmax = 0,15
            # cmap = 'Purples'
            cmap = 'viridis'
            # cmap = 'magma'
            # cmap = 'Greens'
        else:
            vmin,vmax = -5,5
            cmap = 'bwr'
            # cmap = 'PiYG'
        im = ax.imshow(fracmat[:,:,idir,icat],vmin=vmin,vmax=vmax,cmap=cmap)

        ax.set_xticks([0,1,2],['Sub','None','Add'])
        # ax.set_yticks([0,1,2],['Div','None','Mult'])
        ax.set_yticks([0,1,2],['Mult','None','Div'])
        ax.set_xlabel('Addition')
        if icat == 0:
            ax.set_ylabel('Multiplicative')
        ax.set_title(legendlabels[icat] if icat < 2 else 'Diff (%s-%s)' % (legendlabels[1],legendlabels[0]))
        for i in range(3):
            for j in range(3):
                if icat != 2:
                    ax.text(j,i,'%2.1f%%' % fracmat[i,j,idir,icat],ha='center',va='center',color='white' if fracmat[i,j,idir,icat]<20 else 'black')
                else: 
                    ax.text(j,i,'%s%2.1f%%\n%s' % ('+' if fracmat[i,j,idir,icat]>0 else '',fracmat[i,j,idir,icat],get_sig_asterisks(testmat[i,j,idir])),
                            ha='center',va='center',color='black')
    fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04,label='% sign. cells')
plt.tight_layout()
# sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Affine_sig_mod_FF_FB_heatmap_%dsessions' % (nSessions))



#%% Show mean tuned responses and modulation between looped and nonlooped neurons: 
fig,axes = plt.subplots(2,4,figsize=(10,3),sharex=True,sharey=True)
clrs = ['black','red']
axtitles = np.array([['FF PMunl: -corr','FF PMlab: -corr', 'FB V1unl: -corr','FB V1lab: -corr'],
                     ['FF PMunl: +corr','FF PMlab: +corr', 'FB V1unl: +corr','FB V1lab: +corr']])

for ialp,alp in enumerate(arealabelpairs):
    for isign,sign in enumerate([-1,1]):
        ax = axes[isign,ialp]
        idx_N = np.all((
                        corrsig_cells[ialp,:]==sign,
                        celldata['nearby'],
                        # celldata['noise_level']<maxnoiselevel,
                        rangeresp>params['minrangeresp'],
                        ),axis=0)
        meandata = np.nanmean(mean_resp_split_aligned[ialp,:,0,idx_N],axis=0)
        ax.plot(oris,meandata,color=clrs[0],alpha=1)
        meandata = np.nanmean(mean_resp_split_aligned[ialp,:,1,idx_N],axis=0)
        ax.plot(oris,meandata,color=clrs[1],alpha=1)
        ax.set_title(axtitles[isign,ialp])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_Looped_Correlations_bootstrapped_%dsessions' % (nSessions), formats = ['png'])









#%% Bootstrapped comparison of correlations and significant correlations with other area: 

# For each bootstrap, for each labeled cell a random nonlabeled cell that is nearby is sampled. 
# This results in an equal number of nonlabeled cells in a paired way. 
# The distribution of correlations is compared to the loop correlation distribution.
# The fraction of significantly positive and negative as well. 
radius          = 50

nboots          = 10

loopfrac        = np.full((2,3),np.nan) # FF vs FB, +corr vs -corr vs modulated
loopmean        = np.full((2),np.nan) # FF vs FB, +corr vs -corr
loopmean_abs    = np.full((2),np.nan) # FF vs FB, +corr vs -corr

bootfrac        = np.full((2,3,nboots),np.nan) # FF vs FB, +corr vs -corr
bootmean        = np.full((2,nboots),np.nan) # FF vs FB, +corr vs -corr
bootmean_abs    = np.full((2,nboots),np.nan) # FF vs FB, +corr vs -corr

binedges        = np.linspace(-1,1,50)
nhistbins       = len(binedges)-1
loophist        = np.full((2,nhistbins),np.nan) # FF vs FB, +corr vs -corr
boothist        = np.full((2,nhistbins,nboots),np.nan) # FF vs FB, +corr vs -corr

idx_N           = np.all((
                    # celldata['noise_level']<params['maxnoiselevel'],
                    rangeresp>params['minrangeresp'],
                    ),axis=0)

idx_PMlab       = np.where(celldata['arealayerlabel'] == 'PMlabL2/3')[0]
idx_V1lab       = np.where(celldata['arealayerlabel'] == 'V1labL2/3')[0]
idx_PMlab_hasnearby = np.full(len(idx_PMlab),False)
idx_V1lab_hasnearby = np.full(len(idx_V1lab),False)

for iboot in tqdm(range(nboots),total=nboots,desc='Bootstrapping'):
    idx_PMlab_nearby = np.full(len(idx_PMlab),np.nan)
    for iN,N in enumerate(idx_PMlab):
        #get index of which session this labeled cell comes from:
        ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0] 
        #get index of all cells in this session
        idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
        #get index of labeled cell in this session
        idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
        #get index of all unlabeled cells in this session that are nearby this particular labeled cell
        idx_nearby_ses = np.where(np.all((np.squeeze(sessions[ises].distmat_xyz[idx_N_ses,:]<radius),
                                                 rangeresp[idx_ses]>params['minrangeresp'],
                                                 sessions[ises].celldata['redcell']==0,
                                                #  sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                 ),axis=0))[0]
        
        #Convert this index to the index in the whole dataset
        idx_nearby = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
        if len(idx_nearby) > 0: #pick a random one from the selected nearby cells
            idx_PMlab_nearby[iN] = np.random.choice(idx_nearby,1)
            idx_PMlab_hasnearby[iN] = True
    if iboot == 0: 
        print('PMlab: %d/%d' % (np.sum(~np.isnan(idx_PMlab_nearby)),len(idx_PMlab)))
    idx_PMlab_nearby = idx_PMlab_nearby[~np.isnan(idx_PMlab_nearby)].astype(int) #remove nans

    bootfrac[0,0,iboot] = np.sum(corrsig_cells[0,idx_PMlab_nearby]==1) / len(idx_PMlab_nearby) #compute fraction of sig pos for this boot
    bootfrac[0,1,iboot] = np.sum(corrsig_cells[0,idx_PMlab_nearby]==-1) / len(idx_PMlab_nearby)
    
    histcounts = np.histogram(corrdata_cells[0,idx_PMlab_nearby],bins=binedges)[0]
    boothist[0,:,iboot] = np.cumsum(histcounts)/np.sum(histcounts)
    bootmean[0,iboot] = np.nanmean(corrdata_cells[0,idx_PMlab_nearby])
    bootmean_abs[0,iboot] = np.nanmean(np.abs(corrdata_cells[0,idx_PMlab_nearby]))

    idx_V1lab_nearby = np.full(len(idx_V1lab),np.nan)
    for iN,N in enumerate(idx_V1lab):
        ises        = np.where(np.isin(sessiondata['session_id'],celldata['session_id'][N]))[0][0]
        idx_ses     = np.isin(celldata['session_id'],sessions[ises].session_id)
        idx_N_ses   = np.where(np.isin(sessions[ises].celldata['cell_id'],celldata['cell_id'][N]))[0]
        
        idx_nearby_ses = np.where(np.all((np.squeeze(sessions[ises].distmat_xyz[idx_N_ses,:]<radius),
                                                 rangeresp[idx_ses]>params['minrangeresp'],
                                                 sessions[ises].celldata['redcell']==0,
                                                #  sessions[ises].celldata['noise_level']<maxnoiselevel,
                                                 ),axis=0))[0]
        idx_nearby = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]

        if len(idx_nearby) > 0:
            idx_V1lab_nearby[iN] = np.random.choice(idx_nearby,1)
            idx_V1lab_hasnearby[iN] = True
    if iboot == 0: 
        print('V1lab: %d/%d' % (np.sum(~np.isnan(idx_V1lab_nearby)),len(idx_V1lab_nearby)))
    idx_V1lab_nearby = idx_V1lab_nearby[~np.isnan(idx_V1lab_nearby)].astype(int)

    bootfrac[1,0,iboot] = np.sum(corrsig_cells[2,idx_V1lab_nearby]==1) / len(idx_V1lab_nearby)
    bootfrac[1,1,iboot] = np.sum(corrsig_cells[2,idx_V1lab_nearby]==-1) / len(idx_V1lab_nearby)
    bootfrac[:,2,:] = (bootfrac[:,0,:]+bootfrac[:,1,:])  

    histcounts = np.histogram(corrdata_cells[2,idx_V1lab_nearby],bins=binedges)[0]
    boothist[1,:,iboot] = np.cumsum(histcounts)/np.sum(histcounts)
    bootmean[1,iboot] = np.nanmean(corrdata_cells[2,idx_V1lab_nearby])
    bootmean_abs[1,iboot] = np.nanmean(np.abs(corrdata_cells[2,idx_V1lab_nearby]))

#%% Now calculated actual looped data:
idx_N           = np.all((
                    rangeresp>params['minrangeresp'],
                    # np.logical_or(np.isin(range(nCells),idx_V1lab[idx_V1lab_hasnearby]),
                                #   np.isin(range(nCells),idx_PMlab[idx_PMlab_hasnearby])),
                    ),axis=0)
# print(np.sum(idx_N))
loopfrac[0,0]   = np.sum(corrsig_cells[1,idx_N]==1) / np.sum(~np.isnan(corrsig_cells[1,idx_N]))
loopfrac[0,1]   = np.sum(corrsig_cells[1,idx_N]==-1) / np.sum(~np.isnan(corrsig_cells[1,idx_N]))
loopfrac[1,0]   = np.sum(corrsig_cells[3,idx_N]==1) / np.sum(~np.isnan(corrsig_cells[3,idx_N]))
loopfrac[1,1]   = np.sum(corrsig_cells[3,idx_N]==-1) / np.sum(~np.isnan(corrsig_cells[3,idx_N]))
loopfrac[:,2]   = loopfrac[:,0] + loopfrac[:,1]

loopmean[0]     = np.nanmean(corrdata_cells[1,idx_N])
loopmean[1]     = np.nanmean(corrdata_cells[3,idx_N])
loopmean_abs[0] = np.nanmean(np.abs(corrdata_cells[1,idx_N]))
loopmean_abs[1] = np.nanmean(np.abs(corrdata_cells[3,idx_N]))

histcounts      = np.histogram(corrdata_cells[1,idx_N],bins=binedges)[0]
loophist[0,:]   = np.cumsum(histcounts)/np.sum(histcounts)
histcounts      = np.histogram(corrdata_cells[3,idx_N],bins=binedges)[0]
loophist[1,:]   = np.cumsum(histcounts)/np.sum(histcounts)

#%% 
legendlabels        = ['FF','FB']
axisbuffer          = 0.025
lw                  = 2

# subplotlabels = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.'])
# loopdata_subplots = np.stack((loopmean,loopmean_abs,loopfrac[:,0],loopfrac[:,1]),axis=1)
# bootdata_subplots = np.stack((bootmean,bootmean_abs,bootfrac[:,0],bootfrac[:,1]),axis=1)
subplotlabels = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.','Frac. Mod.'])
loopdata_subplots = np.stack((loopmean,loopmean_abs,loopfrac[:,0],loopfrac[:,1],loopfrac[:,2]),axis=1)
bootdata_subplots = np.stack((bootmean,bootmean_abs,bootfrac[:,0],bootfrac[:,1],bootfrac[:,2]),axis=1)
nmetrics = len(subplotlabels)

fig,axes = plt.subplots(2,nmetrics+1,figsize=(nmetrics*2.2,4))

for ialp in range(2):
    axes[ialp,0].plot(binedges[:-1],loophist[ialp,:],color=clrs_arealabelpairs[ialp])
    
    shaded_error(binedges[:-1],np.nanmean(boothist[ialp,:,:],axis=1),np.nanstd(boothist[ialp,:,:],axis=1),
                    ax=axes[ialp,0],color='grey')
    axes[ialp,0].set_xlim([binedges[np.where(loophist[ialp,:]>0)[0][0]],binedges[np.where(loophist[ialp,:]>0.999)[0][0]]])
    axes[ialp,0].set_ylim([0,1])
    axes[ialp,0].set_ylabel(legendlabels[ialp],fontsize=15,fontweight='bold',color=clrs_arealabelpairs[ialp])
    if ialp == 0:
        axes[ialp,0].set_title('Corr. coeff.')
    
    for imetric in range(nmetrics):
        axidx = imetric+1
        axes[ialp,axidx].axvline(loopdata_subplots[ialp,imetric],color=clrs_arealabelpairs[ialp],linewidth=lw)
        sns.histplot(bootdata_subplots[ialp,imetric,:],ax=axes[ialp,axidx],bins=np.linspace(-.1,1,500),element='step',stat='probability',color='grey')
        axes[ialp,axidx].set_xlim([np.percentile(bootdata_subplots[ialp,imetric],0)-axisbuffer,
                                   np.percentile(bootdata_subplots[ialp,imetric],100)+axisbuffer])
        pval = np.sum(bootdata_subplots[ialp,imetric,:]>loopdata_subplots[ialp,imetric])/len(bootdata_subplots[ialp,imetric,:])
        axes[ialp,axidx].text(loopdata_subplots[ialp,imetric],0.1,get_sig_asterisks(np.min([pval,1-pval]),return_ns=True),fontsize=16,color=clrs_arealabelpairs[ialp])
        if ialp == 0:
            axes[ialp,axidx].set_title(subplotlabels[imetric])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'FF_FB_Looped_Correlations_bootstrapped_%dsessions' % (nSessions), formats = ['png'])

#%% 




