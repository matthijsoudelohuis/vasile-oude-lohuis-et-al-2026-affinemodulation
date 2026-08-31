# -*- coding: utf-8 -*-
"""
This script analyzes responses to visual gratings in a multi-area calcium imaging
dataset with labeled projection neurons. The visual stimuli are oriented gratings.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% ###################################################
import math, os
os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from loaddata.get_data_folder import get_local_drive

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from loaddata.session_info import filter_sessions,load_sessions
from utils.tuning import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.explorefigs import plot_excerpt,plot_PCA_gratings,plot_tuned_response
from params import load_params

#%% Plotting and parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches
params  = load_params()

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\ExampleTraces\\')

#%% Load an example session: 
# session_list        = np.array(['LPE12223_2024_06_10']) #GR
session_list        = np.array([['LPE11086_2024_01_05']])

sessions,nSessions   = filter_sessions(protocols = 'GR',only_session_id=session_list,
                                       filter_areas=['V1','PM'])

#%%  Load data properly:        

for ises in range(nSessions):
    sessions[ises].load_tensor(load_calciumdata=True,calciumversion=params['calciumversion'],keepraw=True,
                               load_behaviordata=True,load_videodata=True)
t_axis = sessions[0].t_axis

#%% compute tuning metrics:
idx_resp = (t_axis>=0) & (t_axis<=1)
sessions[0].respmat = np.nanmean(sessions[0].tensor[:,:,idx_resp],axis=2)
sessions = compute_tuning_wrapper(sessions)

#%% Concatenate celldata across sessions:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

#%% Select example cells and trials for plotting:
# np.random.seed(0)
example_cells   = np.array([],dtype=int)
n_example_cells = 12
example_cells=  np.array([ 287, 339, 611,154 ,898 , 374  ,996 ,936 ,1077,1030,1081,1338  ])
# trialsel        = (100,125)
trialsel        = (1490,1520)

if not example_cells.any(): 
    for arealabel in ['PMlab']:
        idx_N = np.all((sessions[0].celldata['arealabel']==arealabel,
                sessions[0].celldata['tuning_var'] > np.percentile(sessions[0].celldata['tuning_var'],80),
                # sessions[0].celldata['noise_level'] < np.percentile(sessions[0].celldata['noise_level'],40)
                ),axis=0)
        idx_N_sub = np.random.choice(np.where(idx_N)[0],n_example_cells//4,replace=False)
        example_cells = np.append(example_cells,idx_N_sub)
    print('Example cells: %s' % example_cells)

#%% Make figure:
fig = plot_excerpt(sessions[0],trialsel=trialsel,neural_version='traces',neuronsel=example_cells,
                   plot_behavioral=False)
ax  = fig.axes[0]
leg = ax.get_legend()
leg.set_title('stimulus direction ($^\circ$)')
leg.set_bbox_to_anchor((1.75, 0.5))
my_savefig(fig,savedir,'Excerpt_GR_%s' % (sessions[0].session_id))

#%%
n_example_cells = 10
example_cells = np.random.choice(np.where(
    sessions[0].celldata['tuning_var'] > np.percentile(sessions[0].celldata['tuning_var'],90))[0],n_example_cells,replace=False)
fig = plot_tuned_response(sessions[0].tensor,sessions[0].trialdata,t_axis,example_cells,plot_n_trials=10)
fig.suptitle('%s' % sessions[0].session_id,fontsize=12)
# save the figure
my_savefig(fig,savedir,'TunedResponse_ExampleTrials_%s' % sessions[0].session_id)

