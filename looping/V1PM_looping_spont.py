#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress,ranksums
from scipy import stats

os.chdir('c:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.session_info import *
from loaddata.get_data_folder import get_local_drive
from utils.gain_lib import * 
from utils.pair_lib import *
from utils.plot_lib import * #get all the fixed color schemes

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\Spontaneous\\')

#%% Plotting and parameters:
params  = load_params()
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])
# session_list            = np.array([['LPE09665_2023_03_21']])

sessions,nSessions      = filter_sessions(protocols = ['SP'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all sessions: 
sessions,nSessions   = filter_sessions(protocols = 'SP')
sessiondata          = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:
for ises in range(nSessions):
    sessions[ises].load_data(load_calciumdata=True,
                             calciumversion=params['calciumversion'])
    sessions[ises].calciumdata /= sessions[ises].celldata['meanF'].to_numpy()[None,:] #convert to deconv/F0

# from utils.filter_lib import my_highpass_filter
# from scipy.signal import detrend

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

nCells                  = len(celldata)

#criteria for selecting still trials:
params['alpha_crossrate'] = 0.0001

#Correlation output:
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)
pvals = np.full((narealabelpairs,nSessions),np.nan)
for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    # [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    # respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    # respdata            = sessions[ises].calciumdata.to_numpy().T #cells x timepoints
    respdata            = sessions[ises].calciumdata.T #cells x timepoints
    # idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
    #                         sessions[ises].respmat_runspeed < params['maxrunspeed'])
    
    for ialp,alp in enumerate(arealabelpairs):
        
        idx_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                                    sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]
        idx_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                                    sessions[ises].celldata['nearby']
                                                    ),axis=0))[0]

        # idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        # idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])

        if len(idx_N1) < params['minnneurons'] or len(idx_N2) < params['minnneurons'] or len(idx_N3) < params['minnneurons']:
            continue

        if params['activitymetric'] == 'mean':#Just mean activity:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        elif params['activitymetric'] == 'ratio': #Ratio:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        elif params['activitymetric'] == 'difference': #Difference:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        
        corrdata_cells[ialp,idx_ses] = tempcorr

        # from statsmodels.stats.multitest import multipletests
        # _,tempsig,_,_ = multipletests(tempsig, alpha=0.05, method='holm', maxiter=1, is_sorted=False, returnsorted=False)
        # tempsig = (tempsig<0.05) * np.sign(tempcorr)
        
        # tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        # tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])

        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<params['alpha_crossrate']) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

        # tempsig = (tempsig<params['alpha_crossrate']) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

#%% 
######  ####### ####### #######    ######     #    ######  ### #     #  #####  
#     # #     # #     #    #       #     #   # #   #     #  #  #     # #     # 
#     # #     # #     #    #       #     #  #   #  #     #  #  #     # #       
######  #     # #     #    #       ######  #     # #     #  #  #     #  #####  
#     # #     # #     #    #       #   #   ####### #     #  #  #     #       # 
#     # #     # #     #    #       #    #  #     # #     #  #  #     # #     # 
######  ####### #######    #       #     # #     # ######  ###  #####   #####  

#%% Bootstrapped comparison of correlations and significant correlations with other area: 
# The distribution of correlations is compared to the loop correlation distribution.
# The fraction of significantly positive and negative as well. 
radii           = np.arange(30,200,20)
nradii          = len(radii)

idx_PMlab       = np.where(celldata['arealayerlabel'] == 'PMlabL2/3')[0]
idx_V1lab       = np.where(celldata['arealayerlabel'] == 'V1labL2/3')[0]

idx_PMlab       = np.where(np.all((
                                    celldata['arealayerlabel'] == 'PMlabL2/3',
                                    celldata['noise_level']<params['maxnoiselevel'],
                                    ),axis=0))[0]
idx_V1lab       = np.where(np.all((
                                    celldata['arealayerlabel'] == 'V1labL2/3',
                                    celldata['noise_level']<params['maxnoiselevel'],
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
                                        sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        # sessions[ises].celldata['arealayerlabel'] == 'PMunlL2/3',
                                        ),axis=0))[0]
    
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
                                        # sessions[ises].celldata['arealayerlabel'] == 'V1unlL2/3',
                                        sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        ),axis=0))[0]
    idx_nearby_all = np.where(np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_nearby_ses]))[0]
    idx_V1lab_allnear[iN,0,:len(idx_nearby_ses)] = idx_nearby_all
    idx_V1lab_allnear[iN,1,:len(idx_nearby_ses)] = distmat[np.ix_(idx_N_ses,idx_nearby_ses)]

#%% 
nboots          = 100
# nboots          = 1000

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
# my_savefig(fig,savedir,'Looped_Modulations_Spontaneous_Bootstrap_Radii')


#%% Plotting bootstrapped results at a specific radius:
radius              = 50
irad                = np.where(radii==radius)[0][0]

legendlabels        = ['FF','FB']
axisbuffer          = 0.005
lw                  = 2
nbins               = 30
subplotlabels       = np.array(['Mean','Abs. Mean','Frac. Pos.','Frac. Neg.','Frac. Mod.'])
loopdata_subplots   = np.stack((loopmean[:,irad],loopmean_abs[:,irad],loopfrac[:,0,irad],loopfrac[:,1,irad],loopfrac[:,2,irad]),axis=1)
bootdata_subplots   = np.stack((bootmean[:,irad],bootmean_abs[:,irad],bootfrac[:,0,irad],bootfrac[:,1,irad],bootfrac[:,2,irad]),axis=1)
nmetrics            = len(subplotlabels)

fig,axes = plt.subplots(2,nmetrics+1,figsize=(nmetrics*2.2,4),sharex='col')

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
        xlims = [np.min([np.percentile(bootdata_subplots[:,imetric,:],0),np.max(loopdata_subplots[:,imetric])])-axisbuffer,
                 np.max([np.percentile(bootdata_subplots[:,imetric,:],100),np.max(loopdata_subplots[:,imetric])])+axisbuffer]
     
        # xlims = [np.min([np.percentile(bootdata_subplots[ialp,imetric],0),loopdata_subplots[ialp,imetric]])-axisbuffer,
                #  np.max([np.percentile(bootdata_subplots[ialp,imetric],100),loopdata_subplots[ialp,imetric]])+axisbuffer]
        ax.set_xlim(xlims) #set lim to extremes of bootstrapped data + small buffer
        pval = np.sum(bootdata_subplots[ialp,imetric,:]>loopdata_subplots[ialp,imetric])/len(bootdata_subplots[ialp,imetric,:])
        ax.text(loopdata_subplots[ialp,imetric],ax.get_ylim()[1]*0.8,get_sig_asterisks(np.min([pval,1-pval]),return_ns=True),fontsize=12,color=clrs_arealabelpairs[ialp])
        if ialp == 0:
            ax.set_title(subplotlabels[imetric])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'Looped_Modulations_Spontaneous_Bootstrap_Radius%d' % (radius))

#%% 






