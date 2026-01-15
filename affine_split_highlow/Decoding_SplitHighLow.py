#%% Import packages
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
from scipy.stats import linregress
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import stats
from statsmodels.stats.anova import AnovaRM

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from params import load_params
from loaddata.session_info import *
from loaddata.get_data_folder import get_local_drive
from utils.pair_lib import *
from utils.plot_lib import * #get all the fixed color schemes
from utils.regress_lib import *

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\Decoding\\')

#%% Plotting and parameters:
params  = load_params()

# Criteria for selecting still trials:
# params['maxrunspeed'] = 5
# params['maxrunspeed'] = 0.5

set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Load all GR sessions: 
sessions,nSessions      = filter_sessions(protocols = 'GR',filter_noiselevel=True)
for ises in range(nSessions):
    sessions[ises].load_respmat(calciumversion=params['calciumversion'])

#%% Select only cells nearby labeled cells to ensure fair comparison labeled and unlabeled cells:
for ises in range(nSessions):   
    sessions[ises].celldata['nearby'] = filter_nearlabeled(sessions[ises],radius=params['radius'])

#%% Get concatenated data:
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

#%%
######  #######  #####  ####### ######  ### #     #  #####  
#     # #       #     # #     # #     #  #  ##    # #     # 
#     # #       #       #     # #     #  #  # #   # #       
#     # #####   #       #     # #     #  #  #  #  # #  #### 
#     # #       #       #     # #     #  #  #   # # #     # 
#     # #       #     # #     # #     #  #  #    ## #     # 
######  #######  #####  ####### ######  ### #     #  #####  

######  ####### ######      #####  ### ####### ####### 
#     # #     # #     #    #     #  #       #  #       
#     # #     # #     #    #        #      #   #       
######  #     # ######      #####   #     #    #####   
#       #     # #                #  #    #     #       
#       #     # #          #     #  #   #      #       
#       ####### #           #####  ### ####### ####### 

#%% Decoding parameters:
model_name          = 'SVM'
scoring_type        = 'accuracy_score'
kfold               = 5
lam                 = 0.5
nmodelfits          = 10

#%% Baseline decoding: 
arealabelpairs          = ['PMunlL2/3','V1unlL2/3']
legendlabels            = ['PM','V1']

narealabelpairs         = len(arealabelpairs)
clrs_arealabelpairs     = get_clr_areas(legendlabels)

popsizes                = np.array([2,5,10,20,50,100,200])
# popsizes              = np.array([5,10])
npopsizes               = len(popsizes)

error_cv                = np.full((npopsizes,narealabelpairs,nSessions,nmodelfits),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Decoding across sessions'):
# for ises in tqdm([0,1,2],total=nSessions,desc='Decoding across sessions'):
    data            = sessions[ises].respmat
    [N,K]           = np.shape(data)
    ori_ses         = sessions[ises].trialdata['Orientation']

    if params['stilltrialsonly']:
        idx_T = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                            sessions[ises].respmat_runspeed < params['maxrunspeed'])
    else:
        idx_T = np.ones(K,dtype=bool)

    for ialp,alp in enumerate(arealabelpairs):
        idx_N           =  np.where(sessions[ises].celldata['arealayerlabel'] == alp)[0]
        for ipop,popsize in enumerate(popsizes):
            if len(idx_N)>popsize:
                for imf in range(nmodelfits):
                    idx_N_sub = np.random.choice(idx_N,popsize,replace=False)
                    X = data[np.ix_(idx_N_sub,idx_T)].T
                    y = ori_ses[idx_T]
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y.ravel())  # C
                    X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

                    error_cv[ipop,ialp,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                        lam=lam,norm_out=False,subtract_shuffle=False)


#%% Plot decoding performance for different number of V1 and PM neurons

# fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=True,sharey=True)
fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharex=True,sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    ax.scatter(popsizes, np.nanmean(error_cv[:,ialp,:,:],axis=(1,2)), marker='o', color=clrs_arealabelpairs[ialp])
    ax.plot(popsizes,np.nanmean(error_cv[:,ialp,:,:],axis=(2)),color=clrs_arealabelpairs[ialp], linewidth=0.5)
    shaded_error(popsizes,np.nanmean(error_cv[:,ialp,:,:],axis=(2)).T,center='mean',error='sem',color=clrs_arealabelpairs[ialp],ax=ax)
    #Figure make up:
    ax.set_ylim([0,1])
    ax.set_xticks(popsizes[[1,3,4,5,6]])
    ax.axhline(1/16,linestyle='--',color='k',alpha=0.5)
    ax.text(popsizes[-1],1/16+0.1,'Chance',fontsize=7,ha='right',va='center')
    ax.set_xlabel('Population size')
    if ialp == 0:
        ax.set_ylabel('Decoding Performance')
    ax.set_title(legendlabels[ialp],fontsize=9)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
my_savefig(fig,savedir,'Decoding_Perf_PopSize_V1PM_%dsessions' % nSessions)


#%% 
######  #######  #####  ####### ######  ### #     #  #####      #####  ######  #       ### ####### 
#     # #       #     # #     # #     #  #  ##    # #     #    #     # #     # #        #     #    
#     # #       #       #     # #     #  #  # #   # #          #       #     # #        #     #    
#     # #####   #       #     # #     #  #  #  #  # #  ####     #####  ######  #        #     #    
#     # #       #       #     # #     #  #  #   # # #     #          # #       #        #     #    
#     # #       #     # #     # #     #  #  #    ## #     #    #     # #       #        #     #    
######  #######  #####  ####### ######  ### #     #  #####      #####  #       ####### ###    #    


#%% Show tuning curve when activity in the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

clrs_arealabelpairs         = get_clr_arealabelpairs(arealabelpairs)
clrs_arealabels_low_high    = get_clr_area_low_high()  # PMlab-PMunl-V1unl

narealabelpairs         = len(arealabelpairs)

nCells                  = len(celldata)
stims                   = np.sort(sessions[0].trialdata['Orientation'].unique())
nstims                  = len(stims)

mean_resp_split         = np.full((narealabelpairs,nstims,2,nCells),np.nan)
params_regress          = np.full((nCells,narealabelpairs,3),np.nan)

#Correlation output:
corrdata_cells          = np.full((narealabelpairs,nCells),np.nan)
corrsig_cells           = np.full((narealabelpairs,nCells),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing corr rates and affine mod'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    if params['stilltrialsonly']:
        idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                                    sessions[ises].respmat_runspeed < params['maxrunspeed'])
    else:
        idx_T_still = np.ones(K,dtype=bool)
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_N3              = np.where(sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2])[0]

        if len(idx_N1) <  params['minnneurons'] or len(idx_N2) < params['minnneurons']  or len(idx_N3) < params['minnneurons']:
            continue
        
        if params['activitymetric'] == 'mean':#Just mean activity:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        elif params['activitymetric'] == 'ratio': #Ratio:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
        elif params['activitymetric'] == 'difference': #Difference:
            meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,nstims,2])
        errorresp           = np.empty([N,nstims,2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        stims                = np.unique(ori_ses)
        for i,ori in enumerate(stims):
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

        regressdata          = np.full((N,3),np.nan)
        regress_sig          = np.full((N,2),0)
        for n in range(N):
            xdata = meanresp[n,:,0]
            ydata = meanresp[n,:,1]
            regressdata[n,:] = linregress(xdata,ydata)[:3]
        params_regress[idx_ses,ialp,:] = regressdata[idx_N3]

        tempcorr          = np.array([pearsonr(meanpopact,respdata[n,:])[0] for n in idx_N3])
        tempsig          = np.array([pearsonr(meanpopact,respdata[n,:])[1] for n in idx_N3])
        corrdata_cells[ialp,idx_ses] = tempcorr
        tempsig = (tempsig<params['alpha_crossrate']) * np.sign(tempcorr)
        corrsig_cells[ialp,idx_ses] = tempsig

# Compute same metric as Flora:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))


#%% 
def shuf_within_cond(X,y):
    Xshuf = copy.deepcopy(X)
    for icond in range(len(np.unique(y))):
        idx_cond = y == icond
        X_cond = X[idx_cond,:]
        y_cond = y[idx_cond]
        X_cond_shuf = my_shuffle(X_cond,method='random',axis=0)
        Xshuf[idx_cond,:] = X_cond_shuf
    return Xshuf

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

#%% Decoding orientation when the FF or FB activity is high or low:
arealabelpairs  = ['V1lab-V1unl-PMunlL2/3','PMlab-PMunl-V1unlL2/3']
narealabelpairs = len(arealabelpairs)

nsampleneurons      = 50
minnneurons         = 5
nmodelfits          = 25

error_cv            = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)
error_cv_withincondshuf   = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Decoding across sessions'):
# for ises in tqdm([0],total=nSessions,desc='Decoding across sessions'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    data            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    ori_ses         = sessions[ises].trialdata['Orientation']

    if params['stilltrialsonly']:
        # idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < params['maxvideome'],
        #                             sessions[ises].respmat_runspeed < maxrunspeed)
        idx_T_still = np.logical_and(sessions[ises].respmat_videome < params['maxvideome'],
                                    sessions[ises].respmat_runspeed < params['maxrunspeed'])
    else:
        idx_T_still = np.ones(K,dtype=bool)
   
    for ialp,alp in enumerate(arealabelpairs):
        
        for imf in range(nmodelfits):
            idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
            idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

            idx_ses             = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][0])
            idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                          rangeresp[idx_ses]>params['minrangeresp'],
                                          np.any(corrsig_cells[:,idx_ses]==1,axis=0),
                                        #   sessions[ises].celldata['noise_level']<params['maxnoiselevel'],
                                        ),axis=0))[0]

            if len(idx_N1) < minnneurons or len(idx_N2) < minnneurons or len(idx_N3) < nsampleneurons:
                continue

            if lam is None:
                idx_N_sub = np.random.choice(idx_N3,nsampleneurons,replace=False)
                y = ori_ses
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y.ravel())  # Convert to 1D array
                X = data[idx_N_sub,:].T
                X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
                lam = find_optimal_lambda(X,y,model_name=model_name,kfold=kfold)

            if params['activitymetric'] == 'mean':#Just mean activity:
                meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
            elif params['activitymetric'] == 'ratio': #Ratio:
                meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) / np.nanmean(respdata[idx_N2,:],axis=0)
            elif params['activitymetric'] == 'difference': #Difference:
                meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)

            # compute index for trials with low and high activity in the other labeled pop
            idx_K1            = np.array([]).astype(int)
            idx_K2            = np.array([]).astype(int)

            stims                = np.unique(ori_ses)

            for i,ori in enumerate(stims):
                idx_T = np.logical_and(ori_ses == ori,idx_T_still)
                idx_K1 = np.concatenate([idx_K1,
                                        np.where(np.all((ori_ses == ori,
                                                        idx_T_still,
                                                        meanpopact < np.nanpercentile(meanpopact[idx_T],params['splitperc'])),axis=0))[0]])
                idx_K2 = np.concatenate([idx_K2,
                                        np.where(np.all((ori_ses == ori,
                                                        idx_T_still,
                                                        meanpopact > np.nanpercentile(meanpopact[idx_T],100-params['splitperc'])),axis=0))[0]])
            
            idx_N_sub = np.random.choice(idx_N3,nsampleneurons,replace=False)
            X = data[np.ix_(idx_N_sub,idx_K1)].T
            y = ori_ses[idx_K1]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # C
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            error_cv[ialp,0,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)

            Xshuf = shuf_within_cond(X,y)
            error_cv_withincondshuf[ialp,0,ises,imf],_,_,_   = my_decoder_wrapper(Xshuf,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)

            X = data[np.ix_(idx_N_sub,idx_K2)].T
            y = ori_ses[idx_K2]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # C
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            error_cv[ialp,1,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)
            
            Xshuf = shuf_within_cond(X,y)
            error_cv_withincondshuf[ialp,1,ises,imf],_,_,_   = my_decoder_wrapper(Xshuf,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)


#%% Decoding performance increase with increased FF or FB activity:  
legendlabels            = ['FF','FB']

palettes = [['#7BE087','#00E028'],['#DA78E6','#8300E6']]

axtitles = np.array(['Feedforward','Feedback']) 
fig,axes = plt.subplots(1,2,figsize=(5,3),sharey=True)
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    data = np.nanmean(error_cv[ialp,:,:,:],axis=2)
    # data = np.nanmean(error_cv_withincondshuf[ialp,:,:,:],axis=2)

    sns.stripplot(data.T,palette=palettes[ialp],alpha=0.5,ax=ax,jitter=0.1)
    sns.barplot(data.T,palette=palettes[ialp],alpha=0.5,ax=ax)
    sns.lineplot(data,palette=sns.color_palette(['grey'],nSessions),alpha=0.5,ax=ax,legend=False,
                linewidth=2,linestyle='-')

    ax.axhline(1/16,linestyle='--',color='k',alpha=0.5)
    ax.text(0.5,1/16+0.05,'Chance',fontsize=10,ha='center',va='center')
    ax.set_ylim([0,1])
    ax.set_title(axtitles[ialp],fontsize=12,)
    ax.set_xticks([0,1],labels=['Low','High'])
    if ialp == 0:
        ax.set_ylabel('Decoding Performance')
    else:
        ax.set_ylabel('')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)

#Statistics:
testdata    = np.nanmean(error_cv,axis=3) #average over modelfits
# testdata    = np.nanmean(error_cv_withincondshuf,axis=3) #average over modelfits
idx_ses  = ~np.any(np.isnan(testdata),axis=(0,1))
df = pd.DataFrame({'perf': testdata.flatten(),
                   'act': np.repeat(np.tile(np.arange(2),2),nSessions),
                   'area': np.repeat(np.arange(2),2*nSessions),
                   'session_id': np.tile(np.arange(nSessions),2*2)
                   })
df.dropna(inplace=True)
df = df[df['session_id'].isin(np.where(idx_ses)[0])]

# Conduct the repeated measures ANOVA
aov = AnovaRM(data=df,
              depvar='perf',
              subject='session_id',
              within=['act', 'area'])
res = aov.fit()
print(res.summary())

# my_savefig(fig,savedir,'Decoding_Ori_FF_FB_LowHigh_StillOnly_%dsessions' % nSessions, formats = ['png'])

#%% Normalized plot:
# extreme percentiles: 
xoffset         = 1.5
normalize       = True

axtitles = np.array(['Feedforward','Feedback']) 
fig,axes = plt.subplots(1,1,figsize=(4.5*cm,5*cm))
ax = axes
ialp = 0 
data = np.nanmean(error_cv[ialp,:,:,:],axis=2)
if normalize:
    # data -= data[0,:][None,:]
    data /= data[0,:][None,:]
ax.plot(np.tile([0,1],(nSessions,1)).T,data,color='grey',ls='-',linewidth=1)
ax.errorbar(x=[0,1],y=np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(nSessions),color='k',ls='None',linewidth=2)
ax.plot(np.array([0,1]),np.nanmean(data,axis=1),color='black',ls='-',linewidth=2)
add_paired_ttest_results(ax, data[0,:],data[1,:],pos=[0.25,0.5],fontsize=8)

ialp = 1
data = np.nanmean(error_cv[ialp,:,:,:],axis=2)
if normalize:
    # data -= data[0,:][None,:]
    data /= data[0,:][None,:]
    # ydata = error_cv[ialp,1,:,:] / error_cv[ialp,0,:,:]

ax.plot(np.tile([0,1],(nSessions,1)).T+xoffset,data,color='grey',ls='-',linewidth=1)
ax.errorbar(x=np.array([0,1])+xoffset,y=np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(nSessions),
            color='k',ls='None',linewidth=2)
ax.plot(np.array([0,1])+xoffset,np.nanmean(data,axis=1),color='black',ls='-',linewidth=2)
add_paired_ttest_results(ax, data[0,:],data[1,:],pos=[0.75,0.5],fontsize=7)
if not normalize:
    ax.axhline(1/16,linestyle='--',color='k',alpha=0.5)
    ax.text(0.5,1/16+0.05,'Chance',fontsize=8,ha='center',va='center')
    ax.set_ylim([0,1])
else:
    # ax.set_ylim([-0.05,0.2])
    ax.set_ylim([0.9,1.3])
ax.set_ylabel('Relative Decoding Performance',fontsize=7)
ax_nticks(ax,4)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# ax.set_xticks(np.concatenate((np.arange(2),np.arange(2)+xoffset)),labels=['FF Low','FF High', 'FB Low','FB High'])
ax.set_xticks(np.concatenate((np.arange(2),np.arange(2)+xoffset)),labels=['Low','High', 'Low','High'],
        fontsize=6)
ax.set_xlabel('Activity Level',fontsize=7)
ax.text(0.25,1.05,'FF',fontsize=9,ha='center',va='center',transform=plt.gca().transAxes,color=clrs_arealabelpairs[0])
ax.text(0.75,1.05,'FB',fontsize=9,ha='center',va='center',transform=plt.gca().transAxes,color=clrs_arealabelpairs[1])

#Statistics:
testdata    = np.nanmean(error_cv,axis=3) #average over modelfits
idx_ses     = ~np.any(np.isnan(testdata),axis=(0,1)) #select only session for which all conditions are present
df = pd.DataFrame({'perf': testdata.flatten(),
                   'act': np.repeat(np.tile(np.arange(2),2),nSessions),
                   'area': np.repeat(np.arange(2),2*nSessions),
                   'session_id': np.tile(np.arange(nSessions),2*2)
                   })
df.dropna(inplace=True)
df = df[df['session_id'].isin(np.where(idx_ses)[0])]

# Conduct the repeated measures ANOVA
aov = AnovaRM(data=df,
              depvar='perf',
              subject='session_id',
              within=['act', 'area'])
res = aov.fit()
print(res.summary())

restable = res.anova_table
testlabel = ['Activity','Pathway','Interaction']
for i in range(3):
    ax.text(0.5,0.95-0.07*i,'%s%s: F(%d,%d)=%1.2f, p=%1.3f' % (get_sig_asterisks(restable['Pr > F'][i]),testlabel[i],restable['Num DF'][i],restable['Den DF'][i],restable['F Value'][i],restable['Pr > F'][i])
            ,transform=plt.gca().transAxes,fontsize=4,ha='center',va='center')

# my_savefig(fig,savedir,'Decoding_LowHigh_StillOnly_Normalized_%dsessions' % nSessions)

#%% Normalized plot of within condition shuffled data:
# extreme percentiles: 
xoffset         = 1.5
normalize       = True

axtitles = np.array(['Feedforward','Feedback']) 
fig,axes = plt.subplots(1,1,figsize=(4.5*cm,5*cm))
ax = axes
ialp = 0 
data = np.nanmean(error_cv_withincondshuf[ialp,:,:,:],axis=2)
if normalize:
    # data -= data[0,:][None,:]
    data /= data[0,:][None,:]
ax.plot(np.tile([0,1],(nSessions,1)).T,data,color='grey',ls='-',linewidth=1)
ax.errorbar(x=[0,1],y=np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(nSessions),color='k',ls='None',linewidth=2)
ax.plot(np.array([0,1]),np.nanmean(data,axis=1),color='black',ls='-',linewidth=2)
add_paired_ttest_results(ax, data[0,:],data[1,:],pos=[0.25,0.5],fontsize=8)

ialp = 1
data = np.nanmean(error_cv_withincondshuf[ialp,:,:,:],axis=2)
if normalize:
    # data -= data[0,:][None,:]
    data /= data[0,:][None,:]
    # ydata = error_cv_withincondshuf[ialp,1,:,:] / error_cv_withincondshuf[ialp,0,:,:]

ax.plot(np.tile([0,1],(nSessions,1)).T+xoffset,data,color='grey',ls='-',linewidth=1)
ax.errorbar(x=np.array([0,1])+xoffset,y=np.nanmean(data,axis=1),yerr=np.nanstd(data,axis=1)/np.sqrt(nSessions),
            color='k',ls='None',linewidth=2)
ax.plot(np.array([0,1])+xoffset,np.nanmean(data,axis=1),color='black',ls='-',linewidth=2)
add_paired_ttest_results(ax, data[0,:],data[1,:],pos=[0.75,0.5],fontsize=7)
if not normalize:
    ax.axhline(1/16,linestyle='--',color='k',alpha=0.5)
    ax.text(0.5,1/16+0.05,'Chance',fontsize=8,ha='center',va='center')
    ax.set_ylim([0,1])
else:
    # ax.set_ylim([-0.05,0.2])
    ax.set_ylim([0.9,1.3])
ax.set_ylabel('Relative Decoding Performance',fontsize=7)
ax_nticks(ax,4)

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# ax.set_xticks(np.concatenate((np.arange(2),np.arange(2)+xoffset)),labels=['FF Low','FF High', 'FB Low','FB High'])
ax.set_xticks(np.concatenate((np.arange(2),np.arange(2)+xoffset)),labels=['Low','High', 'Low','High'],
        fontsize=6)
ax.set_xlabel('Activity Level',fontsize=7)
ax.text(0.25,1.05,'FF',fontsize=9,ha='center',va='center',transform=plt.gca().transAxes,color=clrs_arealabelpairs[0])
ax.text(0.75,1.05,'FB',fontsize=9,ha='center',va='center',transform=plt.gca().transAxes,color=clrs_arealabelpairs[1])

#Statistics:
testdata    = np.nanmean(error_cv_withincondshuf,axis=3) #average over modelfits
idx_ses     = ~np.any(np.isnan(testdata),axis=(0,1)) #select only session for which all conditions are present
df = pd.DataFrame({'perf': testdata.flatten(),
                   'act': np.repeat(np.tile(np.arange(2),2),nSessions),
                   'area': np.repeat(np.arange(2),2*nSessions),
                   'session_id': np.tile(np.arange(nSessions),2*2)
                   })
df.dropna(inplace=True)
df = df[df['session_id'].isin(np.where(idx_ses)[0])]

# Conduct the repeated measures ANOVA
aov = AnovaRM(data=df,
              depvar='perf',
              subject='session_id',
              within=['act', 'area'])
res = aov.fit()
print(res.summary())

restable = res.anova_table
testlabel = ['Activity','Pathway','Interaction']
for i in range(3):
    ax.text(0.5,0.95-0.07*i,'%s%s: F(%d,%d)=%1.2f, p=%1.3f' % (get_sig_asterisks(restable['Pr > F'][i]),testlabel[i],restable['Num DF'][i],restable['Den DF'][i],restable['F Value'][i],restable['Pr > F'][i])
            ,transform=plt.gca().transAxes,fontsize=4,ha='center',va='center')

my_savefig(fig,savedir,'Decoding_LowHigh_WithinCondShuffle_Normalized_%dsessions' % nSessions)


#%% 
 #####  #     #    #    #       #          ######  ####### ######   #####  
#     # ##   ##   # #   #       #          #     # #     # #     # #     # 
#       # # # #  #   #  #       #          #     # #     # #     # #       
 #####  #  #  # #     # #       #          ######  #     # ######   #####  
      # #     # ####### #       #          #       #     # #             # 
#     # #     # #     # #       #          #       #     # #       #     # 
 #####  #     # #     # ####### #######    #       ####### #        #####  

#%% Relating decoding performance increase to multiplicative gain increases: 

arealabelpairs  = ['V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3']
narealabelpairs         = len(arealabelpairs)

nsampleneurons      = 10
nmodelfits          = 10

error_cv            = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)
params_meansub      = np.full((narealabelpairs,2,nSessions,nmodelfits),np.nan)

for ises in tqdm(range(nSessions),total=nSessions,desc='Decoding across sessions'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    data            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    ori_ses         = sessions[ises].trialdata['Orientation']

    if params['stilltrialsonly']:
        idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < params['maxvideome'],
                                    sessions[ises].respmat_runspeed < maxrunspeed)
    else:
        idx_T_still = np.ones(K,dtype=bool)
   
    for ialp,alp in enumerate(arealabelpairs):
        
    
        idx_N1              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[0])[0]
        
        idx_N2              = np.where(sessions[ises].celldata['arealabel'] == alp.split('-')[1])[0]

        idx_ses             = np.isin(celldata['session_id'],sessions[ises].celldata['session_id'][0])
        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                    #   np.any(corrsig_cells[:,idx_ses]!=1,axis=0),
                                        rangeresp[idx_ses]>params['minrangeresp'],
                                    #   np.any(corrsig_cells[:,idx_ses]==1,axis=0),
                                    ),axis=0))[0]
        
        for imf in range(nmodelfits):
 
            # subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
            # idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
            # idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)

            if len(idx_N1) < minnneurons or len(idx_N2) < minnneurons or len(idx_N3) < nsampleneurons:
                continue

            if params['activitymetric'] == 'mean':#Just mean activity:
                meanpopact          = np.nanmean(data[idx_N1,:],axis=0)
            elif params['activitymetric'] == 'ratio': #Ratio:
                meanpopact          = np.nanmean(data[idx_N1,:],axis=0) / np.nanmean(data[idx_N2,:],axis=0)
            elif params['activitymetric'] == 'difference': #Difference:
                meanpopact          = np.nanmean(data[idx_N1,:],axis=0) - np.nanmean(data[idx_N2,:],axis=0)

            # compute index for trials with low and high activity in the other labeled pop
            idx_K1            = np.array([]).astype(int)
            idx_K2            = np.array([]).astype(int)

            stims                = np.unique(ori_ses)

            for i,ori in enumerate(stims):
                idx_T = np.logical_and(ori_ses == ori,idx_T_still)
                idx_K1 = np.concatenate([idx_K1,
                                        np.where(np.all((ori_ses == ori,
                                                        idx_T_still,
                                                        meanpopact < np.nanpercentile(meanpopact[idx_T],params['splitperc'])),axis=0))[0]])
                idx_K2 = np.concatenate([idx_K2,
                                        np.where(np.all((ori_ses == ori,
                                                        idx_T_still,
                                                        meanpopact > np.nanpercentile(meanpopact[idx_T],100-params['splitperc'])),axis=0))[0]])
            
            idx_N_sub = np.random.choice(idx_N3,nsampleneurons,replace=False)
            X = data[np.ix_(idx_N_sub,idx_K1)].T
            y = ori_ses[idx_K1]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # C
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0

            error_cv[ialp,0,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)

            X = data[np.ix_(idx_N_sub,idx_K2)].T
            y = ori_ses[idx_K2]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y.ravel())  # C
            X,y,_ = prep_Xpredictor(X,y) #zscore, set columns with all nans to 0, set nans to 0
            error_cv[ialp,1,ises,imf],_,_,_   = my_decoder_wrapper(X,y,model_name=model_name,kfold=kfold,scoring_type=scoring_type,
                                                                lam=lam,norm_out=False,subtract_shuffle=False)

            params_regress_ses = params_regress[idx_ses]
            params_meansub[ialp,0,ises,imf] = np.nanmean(params_regress_ses[idx_N_sub,ialp,0])
            params_meansub[ialp,1,ises,imf] = np.nanmean(params_regress_ses[idx_N_sub,ialp,1])

#%% Plot increase in decoding performance for different number of V1 and PM neurons
# fig,axes = plt.subplots(1,2,figsize=(6,3),sharex=False,sharey=True)
fig,axes = plt.subplots(1,2,figsize=(8*cm,4*cm),sharex=False,sharey=True)

for iparam,param in enumerate(['slope','offset']):
    ax = axes[iparam]
    for ialp,alp in enumerate(arealabelpairs):
        xdata = params_meansub[ialp,iparam,:,:].flatten()
        ydata = error_cv[ialp,1,:,:] / error_cv[ialp,0,:,:]
        ydata = ydata.flatten()
        # ax.scatter(xdata,ydata, marker='o', color=clrs_arealabelpairs[ialp])
        # ax.scatter(xdata,ydata, marker='.', color=clrs_arealabelpairs[ialp])
        sns.regplot(x=xdata,y=ydata,ax=ax,scatter=True,ci=95,color=clrs_arealabelpairs[ialp],
                    scatter_kws={'s':1},line_kws={'color':clrs_arealabelpairs[ialp]})
    ax.axhline(0,linestyle='--',color='k',alpha=0.5)

    if iparam==0:
        ax.text(0.8,0.1,legendlabels[0],fontsize=7,ha='center',va='center',transform=ax.transAxes,color=clrs_arealabelpairs[0])
        ax.text(0.8,0.2,legendlabels[1],fontsize=7,ha='center',va='center',transform=ax.transAxes,color=clrs_arealabelpairs[1])
    ax.set_xlabel('Ensemble mean %s' % param)
    if iparam == 0: 
        ax.set_ylabel('Decoding Performance\n (High / Low)',fontsize=6)
    ax.set_title(param,fontsize=8)
    ax_nticks(ax,4)
    ax.set_ylim([0.5,1.75])
    ax.axhline(1,ls=':',linewidth=1,color='k')
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True,offset=3)
# my_savefig(fig,savedir,'DeltaDecoding_Mult_Add_Ensembles_%dsessions' % nSessions)

#%%
