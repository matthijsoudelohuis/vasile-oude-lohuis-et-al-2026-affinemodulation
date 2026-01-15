# -*- coding: utf-8 -*-
"""
This script analyzes the quality of the recordings and their relation 
to various factors such as depth of recording, being labeled etc.
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""
#%% Import the libraries and functions
import os
os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loaddata.session_info import filter_sessions,load_sessions

from sklearn import preprocessing
from utils.plot_lib import *

savedir         = 'E:\\OneDrive\\PostDoc\\Figures\\\Affine_FF_vs_FB\\Labeling\\'

#%% Load the data from all passive protocols:
protocols            = ['GR','GN','IM']
protocols            = ['GR']
# protocols            = ['DN']

sessions,nsessions            = filter_sessions(protocols,min_cells=1)

# session_list        = np.array([['LPE10885','2023_10_23']])
# session_list        = np.array([['LPE11086','2024_01_05']])
# sessions,nsessions  = load_sessions(protocol = 'GR',session_list=session_list)

#%% Set plotting parameters:
cm = 1/2.54  # centimeters in inches
plt.rcParams.update({'font.size': 6, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.titlesize': 8,
                     'axes.labelpad': 1, 'ytick.major.pad': 1, 'xtick.major.pad': 1})

#%% #### reset threshold if necessary:
threshold = 0.5
for ses in sessions:
    ses.reset_label_threshold(threshold)

#%% ######## ############
## Combine cell data from all loaded sessions to one dataframe:
celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

## remove any double cells (for example recorded in both GR and RF)
celldata = celldata.drop_duplicates(subset='cell_id', keep="first")

celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)

celldata.loc[celldata['redcell']==0,'recombinase'] = 'non'

#%% Get information about labeled cells per session per area: 
sesdata = pd.DataFrame()
sesdata['roi_name']         = celldata.groupby(["session_id","roi_name"])['roi_name'].unique()
sesdata['recombinase']      = celldata[celldata['recombinase'].isin(['cre','flp'])].groupby(["session_id","roi_name"])['recombinase'].unique()
sesdata = sesdata.applymap(lambda x: x[0],na_action='ignore')
sesdata['ncells']           = celldata.groupby(["session_id","roi_name"])['nredcells'].count()
sesdata['nredcells']        = celldata.groupby(["session_id","roi_name"])['nredcells'].unique().apply(sum)
sesdata['nlabeled']         = celldata.groupby(["session_id","roi_name"])['redcell'].sum()
sesdata['frac_responsive']  = sesdata['nlabeled'] / sesdata['nredcells'] 
sesdata['frac_labeled']     = sesdata['nlabeled'] / sesdata['ncells'] 

#%% ### Get the number of labeled cells, cre / flp, depth, area etc. for each plane :
planedata = pd.DataFrame()
planedata['depth']          = celldata.groupby(["session_id","plane_idx"])['depth'].unique()
planedata['roi_name']       = celldata.groupby(["session_id","plane_idx"])['roi_name'].unique()
planedata['recombinase']    = celldata[celldata['recombinase'].isin(['cre','flp'])].groupby(["session_id","plane_idx"])['recombinase'].unique()
planedata = planedata.applymap(lambda x: x[0],na_action='ignore')
planedata['ncells']         = celldata.groupby(["session_id","plane_idx"])['depth'].count()
planedata['nlabeled']       = celldata.groupby(["session_id","plane_idx"])['redcell'].sum()
planedata['frac_labeled']   = celldata.groupby(["session_id","plane_idx"])['redcell'].sum() / celldata.groupby(["session_id","plane_idx"])['redcell'].count()
planedata['nredcells']      = celldata.groupby(["session_id","plane_idx"])['nredcells'].mean().astype(int)
planedata['frac_responsive']  = celldata.groupby(["session_id","plane_idx"])['redcell'].sum() / planedata['nredcells'] 


#%% 

#%% ####### Show histogram of ROI overlaps: #######################
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(3.5,3),sharex=True)

sns.histplot(data=celldata,x='frac_red_in_ROI',stat='probability',hue='redcell',
             palette=get_clr_labeled(),binwidth=0.05,ax=ax1)
             
sns.histplot(data=celldata,x='frac_red_in_ROI',stat='probability',hue='redcell',
             palette=get_clr_labeled(),binwidth=0.05,ax=ax2)
fig.subplots_adjust(hspace=0.05)

ax2.get_legend().remove()

ax1.set_xlim([0,1])
ax1.set_ylim([0.8,1])
ax2.set_ylim([0,0.02])

ax1.axvline(threshold,color='grey',linestyle=':')
ax2.axvline(threshold,color='grey',linestyle=':')

ax1.set_xlabel('ROI Overlap')
ax1.set_ylabel('Fraction of cells')
ax2.set_ylabel('')
plt.tight_layout()
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

d = 0.5
kwargs = dict(marker=[(-1,-d),(1,d)],markersize=12,linestyle="none",color='k',mec='k',mew=1,clip_on=False)
ax1.plot([0,1],[0,0],transform=ax1.transAxes,**kwargs)
ax2.plot([0,1],[1,1],transform=ax2.transAxes,**kwargs)

plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Overlap_Dist_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

#%% ####### Show scatter of the two overlap metrics: frac red in ROI and frac of ROI red #######################
fig, ax = plt.subplots(figsize=(3.5,3))
sns.scatterplot(data=celldata,x='frac_red_in_ROI',y='frac_of_ROI_red',hue='redcell',ax=ax,
                palette=get_clr_labeled(),s=5)
ax.get_legend().remove()

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('frac_red_in_ROI')
plt.ylabel('frac_of_ROI_red')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Scatter_Overlap_Twoways_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')




#%% 
areas = ['V1','PM']
clrs_areas          = get_clr_areas(areas)


#%% Scatter plot as a function of depth:
fig, ax = plt.subplots(figsize=(4.5*cm,7*cm))
# sns.scatterplot(data=planedata,x='depth',y='frac_labeled',hue='roi_name',palette=clrs_areas,ax=ax,s=14,hue_order=areas)
sns.scatterplot(data=planedata,y='depth',x='frac_labeled',hue='roi_name',palette=clrs_areas,ax=ax,s=14,hue_order=areas)
sns.lineplot(x=planedata['frac_labeled'],y=planedata['depth'].round(-2),
             hue=planedata['roi_name'],palette=clrs_areas,ax=ax,hue_order=areas,   orient="y")
ax.set_xlabel('Fraction labeled in plane')
ax.set_ylabel(r'Cortical depth ($\mu$m)')
ax.set_ylim([75,500])
ax.invert_yaxis()
ax_nticks(ax,5)
sns.despine(fig=fig, top=True, right=True,offset=3)
plt.tight_layout()
plt.legend(ax.get_legend_handles_labels()[0][:4],areas, loc='best',frameon=False)
my_savefig(fig,savedir,'Frac_labeled_depth_area_%dplanes' % len(planedata))

#%% Bar plot of number of labeled cells per area:
fig, axes = plt.subplots(1,1,figsize=(2.5,2.5))
ax = axes
# ax = axes[0]
sns.barplot(data=sesdata,x='roi_name',y='frac_labeled',palette=clrs_areas,ax=ax,errorbar=None,order=areas,fill=False)
# sns.stripplot(data=sesdata,x='roi_name',y='frac_labeled',color='red',ax=ax,size=6,alpha=0.7,jitter=0.2,order=areas)
sns.stripplot(data=sesdata,x='roi_name',y='frac_labeled',palette=clrs_areas,hue='roi_name',ax=ax,size=6,alpha=0.7,jitter=0.2,
              order=areas,hue_order=areas,legend=False)
# ax.set_title('tdTomato+ fraction')
ax.set_ylabel('tdTomato+ fraction')
ax.set_xlabel(r'Area')
ax_nticks(ax,3)
sns.despine(fig=fig, top=True, right=True, offset=3,trim=False)
plt.tight_layout()

# my_savefig(fig,savedir,'Labeling_Fraction_%danimals' % nanimals)
# plt.savefig(os.path.join(savedir,'Frac_labeled_area_%dsessions' % len(sesdata) + '.png'), format = 'png',bbox_inches='tight')


#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='nredcells',palette=clrs_areas,ax=ax,errorbar='se',order=areas)
sns.stripplot(data=sesdata,x='roi_name',y='nredcells',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2,order=areas)
plt.title('# cellpose cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'nCellpose_area_%dsessions' % len(sesdata) + '.png'), format = 'png',bbox_inches='tight')

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='ncells',palette=clrs_areas,ax=ax,errorbar='se',order=areas)
sns.stripplot(data=sesdata,x='roi_name',y='ncells',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2,order=areas)
plt.title('# suite2p cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'nSuite2p_area_%dsessions' % len(sesdata) + '.png'), format = 'png',bbox_inches='tight')

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='frac_responsive',palette=clrs_areas,ax=ax,errorbar='se',order=areas)
sns.stripplot(data=sesdata,x='roi_name',y='frac_responsive',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2,order=areas)
plt.title('# Frac. responsive cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
# plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Frac_responsive_area_%dsessions' % len(sesdata) + '.png'), format = 'png',bbox_inches='tight')

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='frac_labeled',palette=clrs_areas,ax=ax,errorbar='se',order=areas)
sns.stripplot(data=sesdata,x='roi_name',y='frac_labeled',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2,order=areas)
plt.title('# Frac. labeled cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Frac_labeled_area_%dsessions' % len(sesdata) + '.png'), format = 'png',bbox_inches='tight')

#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=sesdata,x='roi_name',y='nlabeled',palette=clrs_areas,ax=ax,errorbar='se',order=areas)
sns.stripplot(data=sesdata,x='roi_name',y='nlabeled',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2,order=areas)
plt.title('# Labeled cells per session')
plt.ylabel('')
plt.xlabel(r'Area')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'nLabeled_area_%dsessions' % len(sesdata) + '.png'), format = 'png',bbox_inches='tight')


#%% Bar plot of number of labeled cells per area:
fig, ax = plt.subplots(figsize=(3,2.5))
sns.barplot(data=planedata,x='roi_name',y='frac_labeled',palette=clrs_areas,ax=ax,errorbar='se',order=areas)
sns.stripplot(data=planedata,x='roi_name',y='frac_labeled',color='k',ax=ax,size=3,alpha=0.5,jitter=0.2,order=areas)
plt.ylabel('Fraction labeled in plane')
plt.xlabel(r'Area')
plt.tight_layout()
# plt.savefig(os.path.join(savedir,'Frac_labeled_area_%dplanes' % len(planedata) + '.png'), format = 'png',bbox_inches='tight')

#%% Bar plot of difference between cre and flp:
enzymes = ['cre','flp']
clrs_enzymes = get_clr_recombinase(enzymes)
enzymelabels = ['retroAAV-pgk-Cre + \n AAV5-CAG-Flex-tdTomato','retroAAV-EF1a-Flpo + \n AAV1-Ef1a-fDIO-tdTomato']

fig, ax = plt.subplots(figsize=(3,3))
sns.barplot(data=planedata,x='recombinase',y='frac_labeled',palette=clrs_enzymes,ax=ax,errorbar='se')
plt.ylabel('Frac. labeled\n (per plane)')
plt.xlabel(r'Recombinase')
ax.set_xticks([0,1])
ax.set_xticklabels(['\n\nCre','\n\nFlp'], fontsize=8)
ax.set_xticks([0.01,1.01],  minor=True)
ax.set_xticklabels(enzymelabels,fontsize=6, minor=True)

# ax.set_xticklabels(enzymelabels,fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Frac_labeled_enzymes_%dplanes' % len(planedata) + '.png'), format = 'png',bbox_inches='tight')

#%% Scatter plot as a function of depth:
fig, ax = plt.subplots(figsize=(5,4))
sns.scatterplot(data=planedata,x='depth',y='frac_labeled',hue='roi_name',palette=clrs_areas,ax=ax,s=14,hue_order=areas)
plt.ylabel('Fraction labeled in plane')
plt.xlabel(r'Cortical depth ($\mu$m)')
plt.xlim([50,500])
plt.tight_layout()
sns.lineplot(x=planedata['depth'].round(-2),y=planedata['frac_labeled'],
             hue=planedata['roi_name'],palette=clrs_areas,ax=ax,hue_order=areas)
plt.legend(ax.get_legend_handles_labels()[0][:4],areas, loc='best',frameon=False)
# plt.savefig(os.path.join(savedir,'Frac_labeled_depth_area_%dplanes' % len(planedata) + '.png'), format = 'png',bbox_inches='tight')

#%% Number of labeled cells as a function of depth:
fig, ax = plt.subplots(figsize=(5,4))
sns.scatterplot(data=planedata,x='depth',y='nlabeled',hue='roi_name',palette=clrs_areas,ax=ax,s=14,hue_order=areas)
plt.ylabel('Number labeled cells')
plt.xlabel(r'Cortical depth ($\mu$m)')
plt.xlim([50,500])
plt.tight_layout()
sns.lineplot(x=planedata['depth'].round(-2),y=planedata['nlabeled'],
             hue=planedata['roi_name'],palette=clrs_areas,ax=ax,hue_order=areas)
plt.legend(ax.get_legend_handles_labels()[0][:4],areas, loc='best',frameon=False)
plt.savefig(os.path.join(savedir,'NLabeled_depth_area_%dplanes' % len(planedata) + '.png'), format = 'png',bbox_inches='tight')

#%% Number of red cellpose cells as a function of depth (not per se suite2p calcium trace detected):
fig, ax = plt.subplots(figsize=(5,4))
sns.scatterplot(data=planedata,x='depth',y='nredcells',hue='roi_name',palette=clrs_areas,ax=ax,s=14,hue_order=areas)
plt.ylabel('Number labeled in plane')
plt.xlabel(r'Cortical depth ($\mu$m)')
plt.xlim([50,500])
plt.tight_layout()
sns.lineplot(x=planedata['depth'].round(-2),y=planedata['nredcells'],
             hue=planedata['roi_name'],palette=clrs_areas,ax=ax,hue_order=areas)
plt.legend(ax.get_legend_handles_labels()[0][:4],areas, loc='best')
plt.savefig(os.path.join(savedir,'Ncellpose_depth_area_%dplanes' % len(planedata) + '.png'), format = 'png',bbox_inches='tight')

#%% Select only cells nearby labeled cells to ensure fair comparison of quality metrics:
celldata = pd.concat([ses.celldata[filter_nearlabeled(ses,radius=25)] for ses in sessions]).reset_index(drop=True)
# celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
celldata = celldata[celldata['noise_level']<20]

#%% ##################### Cell properties for labeled vs unlabeled cells:
order = [0,1] #for statistical testing purposes
pairs = [(0,1)]
# order = ['non','flp','cre'] #for statistical testing purposes
# pairs = [('non','flp'),('non','cre')]
order = ['unl','lab'] #for statistical testing purposes
pairs = [('unl','lab')]

# fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF","meanF_chan2"]
fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF"]
fields = ["meanF","noise_level","event_rate","skew"]

nfields = len(fields)
fig,axes   = plt.subplots(1,nfields,figsize=(nfields*2,3))

import copy
celldataclip = copy.deepcopy(celldata)

for i in range(nfields):
    ax = axes[i]
    celldataclip[fields[i]] = np.clip(celldata[fields[i]],np.nanpercentile(celldata[fields[i]],0.0),
                                      np.nanpercentile(celldata[fields[i]],99))
    sns.violinplot(data=celldataclip,y=fields[i],x="labeled",palette=['gray','red'],ax=axes[i])
    # sns.violinplot(data=celldata,y=fields[i],x="recombinase",palette=['gray','orangered','indianred'],ax=ax)
    ax.set_ylim(np.nanpercentile(celldataclip[fields[i]],[0.1,99.9]))

    ax.set_xlabel('labeled')
    ax.set_ylabel('')
    ax.set_ylim(0,ax.get_ylim()[1]*1.1)

    annotator = Annotator(ax, pairs, data=celldata, x="labeled", y=fields[i], order=order)
    # annotator = Annotator(ax, pairs, data=celldata, x="recombinase", y=fields[i], order=order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside',verbose=False)
    annotator.apply_and_annotate()
    g  = np.nanmean(celldata.loc[celldata['labeled']=='lab',fields[i]]) /  np.nanmean(celldata.loc[celldata['labeled']=='unl',fields[i]])
    print('{0}: ratio = {1:.1f}%'.format(fields[i],(g-1)*100))
    ax.set_title('%s\n (%+1.1f%%)' % (fields[i],(g-1)*100),fontsize=10) #fields[i])
    # ax.set_title(fields[i])

sns.despine(trim=True,top=True,right=True,offset=3)
# labelcounts = celldata.groupby(['recombinase'])['recombinase'].count()
# plt.suptitle('Quality comparison non-labeled ({0}), cre-labeled ({1}) and flp-labeled ({2}) cells'.format(
    # labelcounts[labelcounts.index=='non'][0],labelcounts[labelcounts.index=='cre'][0],labelcounts[labelcounts.index=='flp'][0]))
plt.tight_layout()
my_savefig(fig,savedir,'Quality_Metrics_%dnearbycells_%dsessions' % (len(celldata),nsessions))


#%% ##################### Cell properties for labeled vs unlabeled cells:
# order = [0,1] #for statistical testing purposes
# pairs = [(0,1)]
order = ['non','flp','cre'] #for statistical testing purposes
pairs = [('non','flp'),('non','cre')]

# fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF","meanF_chan2"]
fields = ["skew","noise_level","event_rate","radius","npix_soma","meanF"]

nfields = len(fields)
fig,axes   = plt.subplots(1,nfields,figsize=(12,4))

for i in range(nfields):
    ax = axes[i]
    # sns.violinplot(data=celldata,y=fields[i],x="redcell",palette=['gray','red'],ax=axes[i])
    sns.violinplot(data=celldata,y=fields[i],x="recombinase",palette=['gray','orangered','indianred'],ax=ax)
    ax.set_ylim(np.nanpercentile(celldata[fields[i]],[0.1,99.9]))

    annotator = Annotator(ax, pairs, data=celldata, x="recombinase", y=fields[i], order=order)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside',verbose=False)
    annotator.apply_and_annotate()

    ax.set_xlabel('labeled')
    ax.set_ylabel('')
    ax.set_title(fields[i])
    # ax.set_ylim(np.nanpercentile(celldata[fields[i]],[0.1,99.8]))

sns.despine(trim=True,top=True,right=True,offset=3)
labelcounts = celldata.groupby(['recombinase'])['recombinase'].count()
plt.suptitle('Quality comparison non-labeled ({0}), cre-labeled ({1}) and flp-labeled ({2}) cells'.format(
    labelcounts[labelcounts.index=='non'][0],labelcounts[labelcounts.index=='cre'][0],labelcounts[labelcounts.index=='flp'][0]))
plt.tight_layout()
# fig.savefig(os.path.join(savedir,'Quality_Metrics_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')
fig.savefig(os.path.join(savedir,'Quality_Metrics_%dnearbycells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

#%% ##################### ###################### ######################
## Scatter of all crosscombinations (seaborn pairplot):
df = celldata[["depth","skew","noise_level","npix_soma",
               "meanF","meanF_chan2","event_rate","redcell"]]
# sns.pairplot(data=df, hue="redcell")

ax = sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap='bwr')
plt.tight_layout()
plt.savefig(os.path.join(savedir,'Quality_Metrics_Heatmap_%dcells_%dsessions' % (len(celldata),nsessions) + '.png'), format = 'png')

#%% ##################################################################