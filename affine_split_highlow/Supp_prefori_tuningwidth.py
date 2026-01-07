#%% 
import os, math
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import linregress
from scipy.optimize import curve_fit

os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

from loaddata.session_info import *
from loaddata.get_data_folder import get_local_drive
from utils.tuning import compute_tuning_wrapper
from utils.plot_lib import * #get all the fixed color schemes

savedir =  os.path.join(get_local_drive(),'OneDrive\\PostDoc\\Figures\\Affine_FF_vs_FB\\SplitTrials\\TuningModulation\\')

#%% #############################################################################
session_list            = np.array([['LPE10919_2023_11_06']])
session_list            = np.array([['LPE12223_2024_06_10']])
session_list            = np.array([['LPE11086_2024_01_05','LPE12223_2024_06_10']])

sessions,nSessions      = filter_sessions(protocols = ['GR'],only_session_id=session_list)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% Load all GR sessions: 
sessions,nSessions   = filter_sessions(protocols = 'GR')

#%%  Load data properly:
calciumversion = 'deconv'
for ises in range(nSessions):
    sessions[ises].load_respmat(load_behaviordata=True, load_calciumdata=True,load_videodata=True,
                                calciumversion=calciumversion,keepraw=False)

#%% Compute Tuning Metrics (gOSI, gDSI etc.)
sessions = compute_tuning_wrapper(sessions)

#%% Re-assign arealayerlabel
for ises in range(nSessions):   
    # sessions[ises].celldata = assign_layer(sessions[ises].celldata)
    sessions[ises].celldata = assign_layer2(sessions[ises].celldata,splitdepth=275)
    sessions[ises].celldata['arealayerlabel'] = sessions[ises].celldata['arealabel'] + sessions[ises].celldata['layer'] 

    sessions[ises].celldata['arealayer'] = sessions[ises].celldata['roi_name'] + sessions[ises].celldata['layer'] 

#%% fit von mises
def vonmises(x,amp,loc,scale):
    return amp * np.exp( (np.cos(x-loc) - 1) / (2 * scale**2) )

def double_vonmises_pi_constrained(x,amp1,amp2,loc,scale,offset):
    return amp1 * np.exp( (np.cos(x-loc) - 1) / (2 * scale**2) ) + amp2 * np.exp( (np.cos(x-loc-np.pi) - 1) / (2 * scale**2) ) + offset

#%% Compute tuning curve when activity in the other area is low or high (only still trials)
arealabelpairs  = [
                    'V1lab-V1unl-PMunlL2/3',
                    'PMlab-PMunl-V1unlL2/3',
                    ]

narealabelpairs         = len(arealabelpairs)

celldata                = pd.concat([sessions[ises].celldata for ises in range(nSessions)]).reset_index(drop=True)

nOris                   = 16
nCells                  = len(celldata)
oris                    = np.sort(sessions[0].trialdata['Orientation'].unique())
perc                    = 25

#criteria for selecting still trials:
maxvideome              = 0.2
maxrunspeed             = 5

minnneurons             = 10
maxnoiselevel           = 20
mean_resp_split         = np.full((narealabelpairs,nOris,2,nCells),np.nan)

#Output double von mises parameters:
param_names           = ['Amplitude (pref)','Amplitude (opposite)',
                         'Preferred Orientation','Tuning Width','Offset']
# nparams                = 4 #amp1, amp2, loc, scale,
nparams                = len(param_names)
VM_data             = np.full((narealabelpairs,nparams,2,nCells),np.nan) #dim1: arealabelpair, dim2: (amp, loc, scale), dim3: low/high, dim4: cell
VM_data_shuffled    = np.full((narealabelpairs,nparams,2,nCells),np.nan) #dim1: arealabelpair, dim2: (amp, loc, scale), dim3: low/high, dim4: cell

for ises in tqdm(range(nSessions),total=nSessions,desc='Computing preferred orientation'):
    [N,K]           = np.shape(sessions[ises].respmat) #get dimensions of response matrix

    respdata            = sessions[ises].respmat / sessions[ises].celldata['meanF'].to_numpy()[:,None]

    idx_T_still = np.logical_and(sessions[ises].respmat_videome/np.nanmax(sessions[ises].respmat_videome) < maxvideome,
                                sessions[ises].respmat_runspeed < maxrunspeed)
    
    for ialp,alp in enumerate(arealabelpairs):
        idx_N1              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[0],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]
        
        idx_N2              = np.where(np.all((
                                    sessions[ises].celldata['arealabel'] == alp.split('-')[1],
                                    sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        idx_N3              = np.where(np.all((sessions[ises].celldata['arealayerlabel'] == alp.split('-')[2],
                                      sessions[ises].celldata['noise_level']<maxnoiselevel,
                                      ),axis=0))[0]

        subsampleneurons = np.min([idx_N1.shape[0],idx_N2.shape[0]])
        idx_N1 = np.random.choice(idx_N1,subsampleneurons,replace=False)
        idx_N2 = np.random.choice(idx_N2,subsampleneurons,replace=False)
        
        idx_ses = np.isin(celldata['cell_id'],sessions[ises].celldata['cell_id'][idx_N3])

        if len(idx_N1) < minnneurons or len(idx_N3) < minnneurons:
            continue
        
        meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0)
        #Difference:
        # meanpopact          = np.nanmean(respdata[idx_N1,:],axis=0) - np.nanmean(respdata[idx_N2,:],axis=0)
        # meanpopact          = np.nanmean(respdata[idx_N2,:],axis=0) - np.nanmean(respdata[idx_N1,:],axis=0)

        # compute meanresp for trials with low and high difference in lab-unl activation
        meanresp            = np.empty([N,len(oris),2])
        meanresp_shuf       = np.empty([N,len(oris),2])
        ori_ses             = sessions[ises].trialdata['Orientation']
        oris                = np.unique(ori_ses)
        for i,ori in enumerate(oris):
            idx_T               = np.logical_and(ori_ses == ori,idx_T_still)
            idx_K1              = meanpopact < np.nanpercentile(meanpopact[idx_T],perc)
            idx_K2              = meanpopact > np.nanpercentile(meanpopact[idx_T],100-perc)
            meanresp[:,i,0]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K1)],axis=1)
            meanresp[:,i,1]     = np.nanmean(respdata[:,np.logical_and(idx_T,idx_K2)],axis=1)

            idx_K1              = np.random.choice(np.where(idx_T)[0],np.sum(np.logical_and(idx_T,idx_K1)),replace=False)
            idx_K2              = np.random.choice(np.where(idx_T)[0],np.sum(np.logical_and(idx_T,idx_K2)),replace=False)
            meanresp_shuf[:,i,0]     = np.nanmean(respdata[:,idx_K1],axis=1)
            meanresp_shuf[:,i,1]     = np.nanmean(respdata[:,idx_K2],axis=1)

        mean_resp_split[ialp,:,:,idx_ses] = meanresp[idx_N3]

        xdata = np.radians(oris)
        # PO_data_ses = np.full((N,2),np.nan)
        # TW_data_ses = np.full((N,2),np.nan)
        # AM_data_ses = np.full((N,2),np.nan)
        VM_data_ses = np.full((N,nparams,2),np.nan)
        VM_data_ses_shuffled = np.full((N,nparams,2),np.nan)
        for iN in range(N):
            try:
                # popt1, pcov = curve_fit(vonmises, xdata, meanresp[iN,:,0],p0=[1,xdata[np.argmax(meanresp[iN,:,0])],0.25])
                popt_low, pcov = curve_fit(double_vonmises_pi_constrained, xdata, meanresp[iN,:,0],p0=[1,1,xdata[np.argmax(meanresp[iN,:,0])],0.25,0])
                # popt_low, pcov = curve_fit(double_vonmises_pi_constrained, xdata, meanresp[iN,:,0],
                #                            p0=[1,1,xdata[np.argmax(meanresp[iN,:,0])],0.25,0],
                #                            bounds=([0,0,0,0,0],[np.inf,np.inf,2*math.pi,np.inf,np.inf]))
                
                
                VM_data_ses[iN,:,0] = popt_low
                popt_low_shuf, pcov = curve_fit(double_vonmises_pi_constrained, xdata, meanresp_shuf[iN,:,0],p0=[1,1,xdata[np.argmax(meanresp_shuf[iN,:,0])],0.25,0])
                VM_data_ses_shuffled[iN,:,0] = popt_low_shuf
            except:
                continue

            try:
                popt_high, pcov = curve_fit(double_vonmises_pi_constrained, xdata, meanresp[iN,:,1],p0=[1,1,xdata[np.argmax(meanresp[iN,:,1])],0.25,0])
                VM_data_ses[iN,:,1] = popt_high
                popt_high_shuf, pcov = curve_fit(double_vonmises_pi_constrained, xdata, meanresp_shuf[iN,:,1],p0=[1,1,xdata[np.argmax(meanresp_shuf[iN,:,1])],0.25,0])
                VM_data_ses_shuffled[iN,:,1] = popt_high_shuf
            except:
                continue

        VM_data[ialp,:,:,idx_ses] = VM_data_ses[idx_N3]
        VM_data_shuffled[ialp,:,:,idx_ses] = VM_data_ses_shuffled[idx_N3]

#%% Convert tuning width from kappa to 1/kappa
VM_data[:,3,:,:] = 1/(VM_data[:,3,:,:]) 
VM_data_shuffled[:,3,:,:] = 1/(VM_data_shuffled[:,3,:,:])

#%%
VM_data[:,2,:,:] = np.mod(np.degrees(VM_data[:,2,:,:]),360)
VM_data_shuffled[:,2,:,:] = np.mod(np.degrees(VM_data_shuffled[:,2,:,:]),360)

#%% Compute response range:
rangeresp = np.nanmax(mean_resp_split,axis=1) - np.nanmin(mean_resp_split,axis=1)
rangeresp = np.nanmax(rangeresp,axis=(0,1))

#%% 
legendlabels = ['FF','FB']
clrs_arealabelpairs = get_clr_arealabelpairs(arealabelpairs)
clrs_arealabels_low_high    = get_clr_area_low_high()  # PMlab-PMunl-V1unl

#%% Show tuning curve when activity in the other area is low or high for an example neuron:
example_cells = [
                    # 'LPE09665_2023_03_21_7_0011', #FF divisive
                    'LPE12223_2024_06_10_3_0023', #FF additive
                    # 'LPE11086_2024_01_05_6_0103', #FF additive
                    # 'LPE09830_2023_04_10_5_0065', #FF additive
                    # 'LPE11086_2024_01_05_4_0002', #FF additive
                    'LPE11086_2024_01_05_5_0169', #FF multiplicative
                    # 'LPE10919_2023_11_06_0_0322', #FF subtractive/divisive
                    # 'LPE11086_2024_01_10_1_0080', #FB additive
                    'LPE11086_2024_01_05_1_0035', #FB additive
                    'LPE12223_2024_06_10_1_0051', #FB multiplicative
                    'LPE11086_2024_01_10_3_0108', #FB multiplicative
                    # 'LPE10885_2023_10_23_1_0276', #FB divisive
                    ]


#%% Plot in two ways:
# example_cells      = [np.random.choice(celldata['cell_id'][idx_examples])]

for example_cell in example_cells:
    idx_N = np.where(celldata['cell_id']==example_cell)[0][0]
    ialp = np.where(~np.isnan(mean_resp_split[:,0,0,idx_N]))[0][0]
    ustim = np.unique(sessions[ises].trialdata['Orientation'])
    xdata = np.radians(oris)
    ylow = mean_resp_split[ialp,:,0,idx_N]
    yhigh = mean_resp_split[ialp,:,1,idx_N]

    fig,axes = plt.subplots(1,1,figsize=(3,2))
    ax = axes
    #Low activity in other area
    ax.scatter(ustim,ylow,color=clrs_arealabels_low_high[ialp,0],s=20)
    popt_low, pcov = curve_fit(double_vonmises_pi_constrained, xdata, ylow,p0=[1,1,xdata[np.argmax(ylow)],0.25,0])
    # popt1, pcov = curve_fit(double_vonmises_pi_constrained, xdata, ylow,p0=[1,1,xdata[np.argmax(ylow)],0.25])
    xfit = np.linspace(0,2*math.pi,100)
    yfit = double_vonmises_pi_constrained(xfit, *popt_low)
    ax.plot(np.degrees(xfit),yfit,color=clrs_arealabels_low_high[ialp,0],linestyle='-',linewidth=0.8)
    #High activity in other area
    ax.scatter(ustim,yhigh,color=clrs_arealabels_low_high[ialp,1],s=20)
    popt_high, pcov = curve_fit(double_vonmises_pi_constrained, xdata, yhigh,p0=[1,1,xdata[np.argmax(yhigh)],0.25,0])
    xfit = np.linspace(0,2*math.pi,100)
    yfit = double_vonmises_pi_constrained(xfit, *popt_high)
    ax.plot(np.degrees(xfit),yfit,color=clrs_arealabels_low_high[ialp,1],linestyle='-',linewidth=0.8)

    ax.set_xlabel('Orientation',fontsize=10)
    ax.set_ylabel('Response',fontsize=10)
    ax.set_xticks([0,90,180,270,360])
    ax.set_yticks([0,my_ceil(np.nanmax([ylow,yhigh]),1)])

    plt.tight_layout()
    sns.despine(fig=fig, top=True, right=True, offset=5,trim=True)
    # my_savefig(fig,savedir,'VonMisesFit_Example_cell_%s' % example_cell)
    # my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell)
    # my_savefig(fig,os.path.join(savedir,'ExampleNeurons','StillOnly','BaselineCorrected'),'FF_FB_affinemodulation_Example_cell_%s' % example_cell, formats = ['png'])


#%% 
histres2D = 10
histres1D = 2.5
# fig,axes = plt.subplots(1,3,figsize=(6,3),sharex=,sharey=True)
fig,axes = plt.subplots(1,3,figsize=(9,3))

PO_data = VM_data[:,1,:,:] #extract preferred orientation

idx_N = rangeresp>0.04
ax = axes[0]
ialp = 0
data = np.degrees(PO_data)
im = ax.hist2d(data[ialp,0,idx_N],data[ialp,1,idx_N],
           bins=np.arange(0,360,histres2D),cmap='magma',density=True,vmax=0.0001)
# bar = plt.colorbar(im[3])
ax.set_yticks(np.arange(0,360,90))
ax.set_xticks(np.arange(0,360,90))
ax.set_xlabel('PO Low')
ax.set_ylabel('PO High')
ax.set_title('FF')

ialp = 1
ax = axes[1]
im = ax.hist2d(data[ialp,0,idx_N],data[ialp,1,idx_N],
           bins=np.arange(0,360,histres2D),cmap='magma',density=True,vmax=0.0001)
# bar = plt.colorbar(im[3])
# ax.set_ylabel('PO High')
ax.set_xlabel('PO Low')

ax.set_title('FB')
ax.set_yticks(np.arange(0,360,90))
ax.set_xticks(np.arange(0,360,90))

PO_drift = np.abs(np.mod(data[:,1,:] - data[:,0,:],360))
ax = axes[2]
sns.histplot(PO_drift[0,idx_N],color='green',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=np.arange(0,190,histres1D))
sns.histplot(PO_drift[1,idx_N],color='purple',element="step",stat="density", 
             common_norm=False,alpha=0.2,ax=ax,bins=np.arange(0,190,histres1D))
# sns.histplot(celldata['PO_drift'][idx_N_high],color='purple',element="step",stat="density", 
#              common_norm=False,alpha=0.2,ax=ax)
# ax.set_xlim([0,25])
ax.legend(['FF','FB'],frameon=False,fontsize=8,title='PO difference')
plt.tight_layout()
# my_savefig(fig,savedir,'PO_drift_FF_FB', formats = ['png'])


#%% Show tuning width for low and high activity in the other area
fig,axes = plt.subplots(1,2,figsize=(4,2.5),sharex=True,sharey=True)
idx_N = rangeresp>0.04
for ialp,alp in enumerate(arealabelpairs):
    ax = axes[ialp]
    data = VM_data[ialp,3,:,:]
    im = ax.scatter(data[0,idx_N],data[1,idx_N],
            color=clrs_arealabelpairs[ialp],alpha=0.2,s=5)
    ax.set_xlabel('TW Low')
    ax.plot([0,1],[0,1],color='grey',linestyle='--',linewidth=0.8,transform=ax.transAxes)
    ax.set_title(legendlabels[ialp])

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3,trim=False)
# my_savefig(fig,savedir,'TuningWidth_FF_FB_Scatter', formats
    # = ['png'])

#%% 
fig,axes = plt.subplots(2,nparams,figsize=(nparams*2,3),sharex='col')
idx_N = rangeresp>0.04
idx_N = np.all((rangeresp>0.04, 
                # VM_data[:,3,0,:]!=0, 
                # VM_data[:,3,1,:]!=0,
                # np.any(VM_data[:,3,:,:]>0, axis=(0,1)),
                # ~np.any(VM_data[:,3,:,:]==0, axis=(0,1)),
                ~np.any(VM_data[:,3,:,:]<=0, axis=(0,1)),
                ~np.any(VM_data[:,3,:,:]>30, axis=(0,1)),
                # np.any(VM_data[:,3,:,:]<30, axis=(0,1)),
                ),axis=0)
trimperc = 2 #trim percentile on each side

for iparam,param in enumerate(param_names):
    for ialp in range(narealabelpairs):
        ax = axes[ialp,iparam]
        
        data = VM_data[ialp,iparam,:,:] #extract preferred orientation
        shufdata = VM_data_shuffled[ialp,iparam,:,:] #extract preferred orientation

        datadiff = data[1,:] - data[0,:]
        shufdatadiff = shufdata[1,:] - shufdata[0,:]

        datadiff = np.mod(datadiff + 180,360) - 180  #wrap to -180 to 180
        shufdatadiff = np.mod(shufdatadiff + 180,360) - 180  #wrap to -180 to 180

        bins = np.linspace(np.nanpercentile(datadiff[idx_N],trimperc),np.nanpercentile(datadiff[idx_N],100-trimperc),20)
        
        sns.histplot(datadiff[idx_N],color=clrs_arealabelpairs[ialp],element="step",stat="count", 
                    common_norm=False,alpha=0.2,ax=ax,bins=bins)
        sns.histplot(shufdatadiff[idx_N],color='grey',element="step",stat="count", 
                    common_norm=False,alpha=0.2,ax=ax,bins=bins,linestyle='dashed')

        if ialp==1: 
            ax.set_xlabel(param)
        if iparam==0: 
            ax.set_ylabel(f'Counts')
        else:
            ax.set_ylabel('')
        if iparam==0:
            ax.text(0.5,0.8,'%s (High-low)' % legendlabels[ialp],color=clrs_arealabelpairs[ialp],transform=ax.transAxes,fontsize=7)
            ax.text(0.5,0.6,'Shuffle',color='grey',transform=ax.transAxes,fontsize=7)

plt.tight_layout()
my_savefig(fig,savedir,'VonMises_TuningParameters_Diff_FF_FB')

