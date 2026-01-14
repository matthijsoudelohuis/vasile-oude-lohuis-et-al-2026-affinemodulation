# -*- coding: utf-8 -*-
"""
This script analyzes responses to visual gratings in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
# os.chdir('c:\\Python\\molanalysis')
os.chdir('c:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import *
from utils.psth import compute_respmat
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import plot_excerpt,plot_PCA_gratings,plot_tuned_response

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\ExampleTraces\\')

#%% Load an example session: 
session_list        = np.array(['LPE12223_2024_06_10']) #GR
session_list        = np.array([['LPE11086_2024_01_05']])

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list,
                                       filter_areas=['V1','PM'])

#%%  Load data properly:        
calciumversion = 'deconv'
# calciumversion = 'dF'
for ises in range(nSessions):
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=calciumversion,keepraw=True,
                               load_behaviordata=True,load_videodata=True)
    # sessions[ises].load_data(load_calciumdata=True,calciumversion=calciumversion,
                            #    load_behaviordata=True,load_videodata=True)
t_axis = sessions[0].t_axis

#%% compute tuning metrics:
idx_resp = (t_axis>=0) & (t_axis<=1)
sessions[0].respmat = np.nanmean(sessions[0].tensor[:,:,idx_resp],axis=2)
sessions = compute_tuning_wrapper(sessions)

#%% Concatenate celldata across sessions:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%%
example_cells   = np.array([],dtype=int)
n_example_cells = 12
trialsel        = (100,125)
trialsel        = (1500,1525)
for arealabel in ['V1unl','V1lab','PMunl','PMlab']:
    idx_N = np.all((sessions[0].celldata['arealabel']==arealabel,
            sessions[0].celldata['tuning_var'] > np.percentile(sessions[0].celldata['tuning_var'],80),
            # sessions[0].celldata['noise_level'] < np.percentile(sessions[0].celldata['noise_level'],40)
            ),axis=0)
    idx_N_sub = np.random.choice(np.where(idx_N)[0],n_example_cells//4,replace=False)
    example_cells = np.append(example_cells,idx_N_sub)

fig = plot_excerpt(sessions[0],trialsel=trialsel,neural_version='traces',neuronsel=example_cells,
                   plot_behavioral=False)
my_savefig(fig,savedir,'Excerpt_GR_%s' % (sessions[0].session_id))

#%%
n_example_cells = 10
example_cells = np.random.choice(np.where(
    sessions[0].celldata['tuning_var'] > np.percentile(sessions[0].celldata['tuning_var'],90))[0],n_example_cells,replace=False)
fig = plot_tuned_response(sessions[0].tensor,sessions[0].trialdata,t_axis,example_cells,plot_n_trials=10)
fig.suptitle('%s - dF/F' % sessions[0].session_id,fontsize=12)
# save the figure
# fig.savefig(os.path.join(savedir,'TunedResponse_dF_%s.png' % sessions[0].session_id))

