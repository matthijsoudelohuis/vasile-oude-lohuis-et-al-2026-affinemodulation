#%% 
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
# from scipy.linalg import norm
# from sklearn.preprocessing import minmax_scale
# from sklearn.metrics import r2_score
# from scipy.stats import linregress,binned_statistic,pearsonr,spearmanr,ks_2samp
# from scipy import stats
# from statsmodels.formula.api import ols

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions
# from utils.tuning import compute_tuning_wrapper,ori_remapping
# from utils.gain_lib import * 
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
    # sessions[ises].load_respmat(calciumversion=params['calciumversion'],keepraw=True)
    sessions[ises].load_data(calciumversion=params['calciumversion'],load_calciumdata=True)

# #%% Compute Tuning Metrics (gOSI, gDSI etc.)
# sessions = ori_remapping(sessions)
# sessions = compute_tuning_wrapper(sessions)

#%% Identify cells near labeled cells
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'])

#%% 
arealabelpairs          = [
                            'V1lab-V1unl',
                            'PMlab-PMunl',
                            ]
narealabelpairs         = len(arealabelpairs)

ises = 7
timewin = [61,81] # seconds
ialp = 0

ises = 2
timewin = [125,145] # seconds
ialp = 1

calciumdata            = np.array(sessions[ises].calciumdata / sessions[ises].celldata['meanF'].to_numpy()[None,:]).T

alp = arealabelpairs[ialp]
fig,ax = plt.subplots(1,1,figsize=(3.8*cm,2.2*cm))
idx_N1              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                            sessions[ises].celldata['nearby']
                                            ),axis=0))[0]
idx_N2              = np.where(np.all((sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                            sessions[ises].celldata['nearby']
                                            ),axis=0))[0]

meanpop1          = np.nanmean(calciumdata[idx_N1,:],axis=0)
meanpop2          = np.nanmean(calciumdata[idx_N2,:],axis=0)
meanpopdiff       = meanpop2 - meanpop1

idx_T = np.where(np.all((sessions[ises].ts_F >= sessions[ises].sessiondata['tStart'][0]+timewin[0],
                            sessions[ises].ts_F <= sessions[ises].sessiondata['tStart'][0]+timewin[1]),axis=0))[0]
ax.plot(sessions[ises].ts_F[idx_T],meanpop1[idx_T],'-',color='red',lw=0.5)
ax.plot(sessions[ises].ts_F[idx_T],meanpop2[idx_T],'-',color='grey',lw=0.5)
# ax.plot(sessions[ises].ts_F,meanpopdiff,'-',color='grey',label='Diff (N=%d)' % (len(idx_N2)))
ax.set_xlim([sessions[ises].sessiondata['tStart'][0]+timewin[0],
                sessions[ises].sessiondata['tStart'][0]+timewin[1]])
ax.legend([arealabelpairs[ialp].split('-')[0],arealabelpairs[ialp].split('-')[1]],fontsize=6,frameon=False)
my_legend_strip(ax)

ax.set_ylabel('Mean ev/F0')
sns.despine(fig,top=True,right=True,offset=2)
my_savefig(fig,savedir,'Mean_LabUnl_%s_%s' % (alp,sessions[ises].session_id))

#%% Compute autocorrelation function
ts = sessions[ises].ts_F
dt = np.median(np.diff(ts))
max_lag_secs = 3.0
max_lag_samps = int(np.round(max_lag_secs / dt))

ac_data = np.zeros((nSessions,narealabelpairs,2*max_lag_samps+1))
lags = np.arange(-max_lag_samps, max_lag_samps + 1) * dt

for ises in range(nSessions):   
    calciumdata            = np.array(sessions[ises].calciumdata / sessions[ises].celldata['meanF'].to_numpy()[None,:])

    for ialp,alp in enumerate(arealabelpairs):
        idx_N1            = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        idx_N2            = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        if (len(idx_N1) < params['minnneurons']) or (len(idx_N2) < params['minnneurons']):
            continue

        meanpop1          = np.nanmean(calciumdata[:,idx_N1],axis=1)
        meanpop2          = np.nanmean(calciumdata[:,idx_N2],axis=1)

        meanpopdiff       = meanpop2 - meanpop1
        # Compute autocorrelation of meanpopdiff over +/-3 seconds
        x = meanpopdiff - np.nanmean(meanpopdiff)
        # replace nan with zero for correlation
        x = np.nan_to_num(x)
        ac_full = np.correlate(x, x, mode='full')
        center = len(ac_full) // 2
        ac_segment = ac_full[center - max_lag_samps:center + max_lag_samps + 1]
        # normalize
        ac_segment = ac_segment / ac_full[center]
        ac_data[ises,ialp,:] = ac_segment
        
#%%
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3*cm,3*cm))

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    # for ises in range(nSessions):
        # ax.plot(lags, ac_data[ises,ialp,:],'-k',lw=0.2)
    # shaded_error(lags, ac_data[:,ialp,:],center='mean',error='std', color='black',
                #  linewidth=0.8,alpha=0.5,ax=ax)
    ax.fill_between(lags, np.nanmean(ac_data[:,ialp,:],axis=0),
                    np.zeros_like(ac_data[:,ialp,:].mean(axis=0)), color='black', alpha=0.5)
    # ax.axvline(0, color='grey', lw=0.25)
    ax.set_xlim(-max_lag_secs, max_lag_secs)
    ax.set_title('%s' % alp,fontsize=6)
    ax.set_xticks([-3,-2,-1,0,1,2,3])
    ax.set_yticks([0,0.5,1])
    ax.set_ylabel('autocorrelation')
    ax.set_xlabel('lag (s)')

sns.despine(fig,top=True,right=True,offset=2,trim=True)
my_savefig(fig,savedir,'Autocorrelation_LabUnl_%dsessions' % (nSessions))




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

#%% 
ises = 2
calciumdata            = np.array(sessions[ises].calciumdata / sessions[ises].celldata['meanF'].to_numpy()[None,:]).T

arealabelpairs          = [
                            'PMlabL2/3-PMunlL2/3',
                            'PMlabL5-PMunlL5',
                            ]
narealabelpairs         = len(arealabelpairs)

timewin = [25,50] # seconds
timewin = [25,50] # seconds
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3.8*cm,2.2*cm),sharey=False)
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    idx_N1              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0],
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]
    idx_N2              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1],
                                                sessions[ises].celldata['nearby']
                                                ),axis=0))[0]

    meanpop1          = np.nanmean(calciumdata[idx_N1,:],axis=0)
    meanpop2          = np.nanmean(calciumdata[idx_N2,:],axis=0)
    meanpopdiff       = meanpop2 - meanpop1

    idx_T = np.where(np.all((sessions[ises].ts_F >= sessions[ises].sessiondata['tStart'][0]+timewin[0],
                                sessions[ises].ts_F <= sessions[ises].sessiondata['tStart'][0]+timewin[1]),axis=0))[0]
    ax.plot(sessions[ises].ts_F[idx_T],meanpop1[idx_T],'-',color='red',lw=0.5)
    ax.plot(sessions[ises].ts_F[idx_T],meanpop2[idx_T],'-',color='grey',lw=0.5)
    # ax.plot(sessions[ises].ts_F,meanpopdiff,'-',color='grey',label='Diff (N=%d)' % (len(idx_N2)))
    ax.set_xlim([sessions[ises].sessiondata['tStart'][0]+timewin[0],
                    sessions[ises].sessiondata['tStart'][0]+timewin[1]])
    ax.legend([arealabelpairs[ialp].split('-')[0],arealabelpairs[ialp].split('-')[1]],fontsize=6,frameon=False)
    my_legend_strip(ax)
    ax.set_ylabel('Mean ev/F0')

sns.despine(fig,top=True,right=True,offset=2)
my_savefig(fig,savedir,'Mean_LabUnl_Layers_%s' % (sessions[ises].session_id))


#%% 
arealabelpairs          = [
                            'PMlabL2/3-PMunlL2/3',
                            'PMlabL5-PMunlL5',
                            ]
narealabelpairs         = len(arealabelpairs)

ts = sessions[ises].ts_F
dt = np.median(np.diff(ts))
max_lag_secs = 3.0
max_lag_samps = int(np.round(max_lag_secs / dt))

ac_data = np.full((nSessions,narealabelpairs,2*max_lag_samps+1), np.nan)
lags = np.arange(-max_lag_samps, max_lag_samps + 1) * dt
corrdata = np.zeros((nSessions))

for ises in range(nSessions):   
    calciumdata            = np.array(sessions[ises].calciumdata / sessions[ises].celldata['meanF'].to_numpy()[None,:])
    meanpopdiff_all = np.zeros((narealabelpairs,calciumdata.shape[0]))

    for ialp,alp in enumerate(arealabelpairs):
    # for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[1])[0]

        if (len(idx_N1) < params['minnneurons']) or (len(idx_N2) < params['minnneurons']):
            continue
        meanpop1          = np.nanmean(calciumdata[:,idx_N1],axis=1)
        meanpop2          = np.nanmean(calciumdata[:,idx_N2],axis=1)

        # meanpop1          = np.nanmean(calciumdata[idx_N1,:],axis=0)
        # meanpop2          = np.nanmean(calciumdata[idx_N2,:],axis=0)
        meanpopdiff       = meanpop2 - meanpop1
        meanpopdiff_all[ialp,:] = meanpopdiff
        
        # Compute autocorrelation of meanpopdiff over +/-3 seconds
        x = meanpopdiff - np.nanmean(meanpopdiff)
        # replace nan with zero for correlation
        x = np.nan_to_num(x)
        ac_full = np.correlate(x, x, mode='full')
        center = len(ac_full) // 2
        ac_segment = ac_full[center - max_lag_samps:center + max_lag_samps + 1]
        # normalize
        ac_segment = ac_segment / ac_full[center]
        ac_data[ises,ialp,:] = ac_segment
    corrdata[ises] = np.corrcoef(meanpopdiff_all[0,:],meanpopdiff_all[1,:])[0,1]

#%% 
print('Correlation between difference metric across the layers: r=%.3f, +-=%.3f (n=%d/12 sessions with min %d labeled neurons in both PM layers)' % (np.nanmean(corrdata),
                                np.nanstd(corrdata),np.sum(~np.isnan(corrdata)), params['minnneurons']))

#%%
fig,axes = plt.subplots(1,narealabelpairs,figsize=(narealabelpairs*3*cm,3*cm),sharey=True)

for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    # for ises in range(nSessions):
        # ax.plot(lags, ac_data[ises,ialp,:],'-k',lw=0.2)
    # shaded_error(lags, ac_data[:,ialp,:],center='mean',error='std', color='black',
                #  linewidth=0.8,alpha=0.5,ax=ax)
    ax.fill_between(lags, np.nanmean(ac_data[:,ialp,:],axis=0),
                    np.zeros_like(ac_data[:,ialp,:].mean(axis=0)), color='black', alpha=0.5)
    ax.set_title('%s' % alp,fontsize=6)
    ax.set_xticks([-3,-2,-1,0,1,2,3])
    ax.set_yticks([0,0.5,1])
    ax.set_xlim(-max_lag_secs, max_lag_secs)
    ax.set_ylabel('autocorrelation')
    ax.set_xlabel('lag (s)')
sns.despine(fig,top=True,right=True,offset=2,trim=True)
my_savefig(fig,savedir,'Autocorrelation_LabUnl_Layers_%dsessions' % (np.sum(~np.isnan(ac_data[:,0,0]))))


#%% 

