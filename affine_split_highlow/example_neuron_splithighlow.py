#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import compute_tuning_wrapper,ori_remapping
from utils.gain_lib import * 
from utils.plot_lib import * #get all the fixed color schemes
from utils.psth import compute_tensor

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
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])
session_list            = np.array([['LPE10919_2023_11_06','LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list,filter_noiselevel=True)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%%  Load data properly:
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=params['calciumversion'],keepraw=True)
    
    [sessions[ises].tensor,t_axis] = compute_tensor(sessions[ises].calciumdata, 
                                                  sessions[ises].ts_F, 
                                                  sessions[ises].trialdata['tOnset'],
                                                  method='nearby')

#%% Get concatenated data:
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

def compute_dprime(X,Y):
    #x and y are vectors of shape (nsamples)
    return (np.nanmean(X) - np.nanmean(Y)) / np.sqrt((np.nanstd(X)**2 + np.nanstd(Y)**2)/2)

def compute_dprime_mat(X,Y):
    #X and Y are matrices of shape (nfeatures,nsamples)
    return (np.nanmean(X,axis=1) - np.nanmean(Y,axis=1)) / np.sqrt((np.nanstd(X,axis=1)**2 + np.nanstd(Y,axis=1)**2)/2)

#%% Show tuning curve when activity in the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())

nTimebins               = len(t_axis)
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells,nTimebins),np.nan)
error_resp_split        = np.full((narealabelpairs,nOris,2,nCells,nTimebins),np.nan)
mean_resp_split_aligned = np.full((narealabelpairs,nOris,2,nCells,nTimebins),np.nan)

dprimedata              = np.full((narealabelpairs,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata        = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]
    tensordata      = sessions[ises].tensor / sessions[ises].celldata['meanF'].to_numpy()[:,None,None]
    idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_N1) < params['minnneurons'] or len(idx_N2) < params['minnneurons']:
            continue

        if params['activitymetric'] == 'mean':#Just mean activity:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        elif params['activitymetric'] == 'ratio': #Ratio:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        elif params['activitymetric'] == 'difference': #Difference:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2,nTimebins])
        errorresp           = np.empty([N,len(oris),2,nTimebins])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            # idx_T               = ori_ses == ori
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)

            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],params['splitperc'])
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-params['splitperc'])
            # meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            # meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)
            # errorresp[:,i,0]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K1)))
            # errorresp[:,i,1]    = np.nanstd(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K2)))

            meanresp[:,i,0,:]     = np.nanmean(tensordata[:,np.logical_and(idx_T,idx_K1),:],axis=1)
            meanresp[:,i,1,:]     = np.nanmean(tensordata[:,np.logical_and(idx_T,idx_K2),:],axis=1)
            errorresp[:,i,0,:]    = np.nanstd(tensordata[:,np.logical_and(idx_T,idx_K1),:],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K1)))
            errorresp[:,i,1,:]    = np.nanstd(tensordata[:,np.logical_and(idx_T,idx_K2),:],axis=1) / np.sqrt(np.sum(np.logical_and(idx_T,idx_K2)))

        meanresp = meanresp - np.nanmin(meanresp[:,:,:],axis=(1,2),keepdims=True)

        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])
        mean_resp_split[ialp,:,:,idx_ses,:] = meanresp[idx_N3]
        error_resp_split[ialp,:,:,idx_ses,:] = errorresp[idx_N3]

        #dprime metric:
        idx_K1              = np.logical_and(meanpopact < np.nanpercentile(meanpopact[idx_T_still],params['splitperc']),
                                             idx_T_still)
        idx_K2              = np.logical_and(meanpopact > np.nanpercentile(meanpopact[idx_T_still],100-params['splitperc']),
                                             idx_T_still)
        
        dprime_ses = compute_dprime_mat(respdata[:,idx_K2],respdata[:,idx_K1])

        dprimedata[ialp,idx_ses] = dprime_ses[idx_N3]

#%% Show some example neurons:
legendlabels        = ['FF','FB']

#%% Plot positively modulated example neurons:
example_cells = [
                    'LPE11086_2024_01_05_5_0018', #FF           #paper FF example 1
                    'LPE11086_2024_01_05_6_0273', #FF           #paper FF example 2

                    'LPE12223_2024_06_10_1_0171', #FB     #paper FB example 1
                    'LPE11086_2024_01_05_0_0062', #FB     #paper FB example 2
                      'LPE11086_2024_01_05_1_0121', #FB
                    ]

#%% Plot negatively modulated example neurons:
example_cells = [
                    'LPE10919_2023_11_06_2_0103', #FF           #paper FF example 1
                    'LPE10919_2023_11_06_0_0140', #FF           #paper FF example 2
                    'LPE10919_2023_11_06_2_0068', #FF           #paper FF example 2

                    'LPE11086_2024_01_05_1_0061', #FB     #paper FB example 1
                    'LPE10919_2023_11_06_4_0016', #FB     #paper FB example 2
                    #   'LPE11086_2024_01_05_1_0121', #FB
                    ]

#%%
ialp = 1
idx_examples = np.all((dprimedata[ialp,:]>np.nanpercentile(dprimedata[ialp,:],90),
                       ),axis=0)

#%%
ialp = 1
idx_examples = np.all((dprimedata[ialp,:]<np.nanpercentile(dprimedata[ialp,:],10),
                       ),axis=0)

#%% Plot the activity over time for example cells:
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]
# istim = 7
# istim = 11
# xanchor = -0.9
xanchor = 2.4
yanchor = 0
clrs_low_high = ['blue','red']
print(example_cells)
for example_cell in example_cells:
    idx_N = np.where(celldata['cell_id']==example_cell)[0][0]
    ialp = np.where(~np.isnan(mean_resp_split[:,0,0,idx_N]))[0][0]
    istim = np.abs(np.diff(np.nanmean(mean_resp_split[ialp,:,:,idx_N,:],axis=-1),axis=1)).argmax()
    ustim = np.unique(sessions[ises].trialdata['Orientation'])
    ymean = mean_resp_split[ialp,istim,:,idx_N,:]
    yerror = error_resp_split[ialp,istim,:,idx_N,:]
    dprime = dprimedata[ialp,idx_N]

    fig,axes = plt.subplots(1,1,figsize=(2.5*cm,2*cm))
    ax = axes
    handles = []
    for iactlevel in range(2):
        handles.append(shaded_error(t_axis,ymean[iactlevel,:],yerror[iactlevel,:],
                                    color=clrs_low_high[iactlevel],alpha=0.5))
                                    # color=clrs_arealabels_low_high[ialp,iactlevel],alpha=0.5))
    ax.legend(handles,['Low','High'],frameon=False,reverse=True if dprime>0 else False) 
    my_legend_strip(ax)
    ax.text(0.2,0.9,r'd`:%s%1.2f' % ('+' if dprime>0 else '',dprime),transform=ax.transAxes)
    ax.plot([0,0.75],[0,0],color='black',ls='-',linewidth=2,solid_capstyle='butt')
    ax.set_xlim([t_axis[0],np.max([t_axis[-1],xanchor+0.1])])
    ax.set_ylim([-0.1,1.1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Activity (dc/dF0)')

    # ax.add_artist(AnchoredSizeBar(ax.transData, 1,
    #               "1 Sec", loc=4, size_vertical=0.03,frameon=False))
    # ax.text(xanchor,yanchor,'1 Sec',transform=ax.transAxes,fontsize=5,ha='left')
    # ax.plot([xanchor,xanchor+1],[yanchor,yanchor],color='black',ls='-',linewidth=2,solid_capstyle='round')
    ax.plot([xanchor,xanchor],[yanchor,yanchor+0.5],color='black',ls='-',linewidth=2,solid_capstyle='butt')
    
    ax.set_axis_off()
    ax_nticks(ax,4)
    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True, offset=2,trim=False)
    # my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly'),'Tensor_modulation_Example_cell_%s' % example_cell)
