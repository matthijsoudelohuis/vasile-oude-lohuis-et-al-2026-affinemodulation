"""
Analyzes stack of GCaMP and tdtomato expressing cells using cellpose software (Pachitariu & Stringer)
Matthijs Oude Lohuis, 2023, Champalimaud Foundation
"""

#%% Import packages
import os
import numpy as np
os.chdir('e:\\Python\\vasile-oude-lohuis-et-al-2026-affinemodulation')

import matplotlib.pyplot as plt
from ScanImageTiffReader import ScanImageTiffReader as imread
from pathlib import Path
from matplotlib.patches import Rectangle
from cellpose import models
# from utils.imagelib import im_norm8
from sklearn.preprocessing import minmax_scale
from utils.plot_lib import *

savedir         = 'E:\\OneDrive\\PostDoc\\Figures\\\Affine_FF_vs_FB\\Labeling\\'
rawdatadir      = 'F:\\Stacks\\'
model_red       = models.CellposeModel(pretrained_model = 'E:\\Python\\cellpose\\models\\redcell_20231107')

rerun           = False # True if you want to rerun cellpose labeling 

#%% Plotting and parameters:
set_plot_basic_config()
cm      = 1/2.54  # centimeters in inches

#%% Directories: 

recomblist = np.array([['NSH07429','flp'], #which recombinase is causing expression in V1, by design other one in PM
['NSH07422','cre'],
['LPE09665','flp'],
['LPE09830','flp'],
['LPE10883','cre'],
['LPE10884','flp'],
['LPE10885','cre'],
['LPE10919','cre'],
['LPE10192','flp'],
['LPE11081','cre'],
['LPE11086','cre'],
['LPE11622','flp'],
['LPE11623','cre'],
['LPE11495','cre'],
['LPE11997','flp'],
['LPE11998','flp'],
['LPE12013','flp'],
['LPE12223','flp'],
['LPE12385','cre']])

#%% Pretrained models to label tdtomato expressing cells:
def get_stack_data_v1(direc,model=model_red):

    nslices         = len(os.listdir(direc))
    assert nslices==75, 'wrong number of slices'

    diam            = 12
    chan            = [[1,0]] # grayscale=0, R=1, G=2, B=3 # channels = [cytoplasm, nucleus]
    nchannels       = 2 #whether image has both red and green channel acquisition (PMT)

    ### Stack loading:
    greenstack  = np.empty([512,512,nslices])
    redstack    = np.empty([512,512,nslices])
    for i,x in enumerate(os.listdir(direc)):
        print(f"Averaging frames for slice {i+1}",end='\r')
        if x.endswith(".tif"):
            fname               = Path(os.path.join(direc,x))
            reader              = imread(str(fname))
            Data                = reader.data()
            greenstack[:,:,i]   = np.average(Data[0::2,:,:], axis=0)
            redstack[:,:,i]     = np.average(Data[1::2,:,:], axis=0)

    # nTdTomCells_V1       = np.zeros(nslices)
    # nTdTomCells_PM       = np.zeros(nslices)
    nTdTomCells       = np.zeros(nslices)
    print('\n')
    ### Get number of tdTomato labeled cells (using cellpose):
    for i in range(nslices):
        print(f"Labeling cells for slice {i+1}",end='\r')
        img_red = np.zeros((512, 512, 3), dtype=np.uint8)
        img_red[:,:,0] = im_norm8(redstack[:,:,i])

        masks_cp_red, flows, styles = model.eval(img_red, diameter=diam, channels=chan)
        nTdTomCells[i]      = len(np.unique(masks_cp_red))-1 #zero is counted as unique

    ### Get the mean fluorescence for each plane:
    meanF2       = [np.mean(redstack[:,:,i]) for i in range(nslices)]
    # meanF1       = [np.mean(greenstack[:,:,i]) for i in range(nslices)]

    data = np.vstack((meanF2,nTdTomCells))

    return data

def get_stack_data_v2(direc,model=model_red):

    nslices         = 75
    nrepeats        = np.min((len(os.listdir(direc)),50))
    # assert nslices==75, 'wrong number of slices'

    diam            = 12
    chan            = [[1,0]] # grayscale=0, R=1, G=2, B=3 # channels = [cytoplasm, nucleus]
    nchannels       = 2 #whether image has both red and green channel acquisition (PMT)

    ### Stack loading:
    greenstack  = np.empty([512,512,nslices,nrepeats])
    redstack    = np.empty([512,512,nslices,nrepeats])
    for i,x in enumerate(os.listdir(direc)[:nrepeats]):
        print(f"Loading data for repeat {i+1}",end='\r')
        if x.endswith(".tif"):
            fname               = Path(os.path.join(direc,x))
            reader              = imread(str(fname))
            Data                = reader.data()
            greenstack[:,:,:,i]   = np.transpose(Data[0::2,:,:],[1,2,0])
            redstack[:,:,:,i]     = np.transpose(Data[1::2,:,:],[1,2,0])

    greenstack   = np.average(greenstack, axis=3)
    redstack     = np.average(redstack, axis=3)

    # nTdTomCells_V1       = np.zeros(nslices)
    # nTdTomCells_PM       = np.zeros(nslices)
    nTdTomCells       = np.zeros(nslices)
    print('\n')
    ### Get number of tdTomato labeled cells (using cellpose):
    for i in range(nslices):
        print(f"Labeling cells for slice {i+1}",end='\r')
        img_red = np.zeros((512, 512, 3), dtype=np.uint8)
        img_red[:,:,0] = im_norm8(redstack[:,:,i])

        masks_cp_red, flows, styles = model.eval(img_red, diameter=diam, channels=chan)
        nTdTomCells[i]      = len(np.unique(masks_cp_red))-1 #zero is counted as unique

    ### Get the mean fluorescence for each plane:
    meanF2       = [np.mean(redstack[:,:,i]) for i in range(nslices)]
    # meanF1       = [np.mean(greenstack[:,:,i]) for i in range(nslices)]

    data = np.vstack((meanF2,nTdTomCells))

    return data


def plot_depthprofile(data,ax,clr):
    # data is assumed to have shape S x M where 
    # S is slices (75) and M is number of mice 
    data_mean = np.nanmean(data,axis=1)
    data_err = np.nanstd(data,axis=1) / np.sqrt(np.shape(data)[1])

    ax.plot(data,slicedepths,c=clr,linewidth=0.25)
    h, = ax.plot(data_mean,slicedepths,c=clr,linewidth=3)
    # h = shaded_error(slicedepths,data.T,center='mean',error='std',color=clr,ax=ax)
    # h, = ax.plot(x,ycenter,color=color,linestyle=linestyle,label=label,linewidth=linewidth)
    # ax.fill_between(x, ycenter-yerror, ycenter+yerror,color=color,alpha=alpha)
    ax.fill_betweenx(slicedepths,data_mean-data_err, data_mean+data_err,color=clr,alpha=0.4)

    return h

def im_norm(I,min=0,max=100):
    mn = np.percentile(I,min)
    mx = np.percentile(I,max)
    mx -= mn
    I = ((I - mn)/mx) * 255
    I[I<0] =0
    I[I>255] = 255
    return I

def im_norm8(I,min=0,max=100):
    I = im_norm(I,min=min,max=max)
    # Specify the data type so that float value will be converted to int 
    return I.astype(np.uint8)

#%% ################################################################
## Get animal information and slice parameters:

animal_ids      = [f.name for f in os.scandir(rawdatadir) if f.is_dir() and f.name.startswith(('LPE','NSH'))]
nanimals        = len(animal_ids)

nslices         = 75
slicedepths     = np.linspace(0,10*(nslices-1),nslices)

#%% ################################################################
## Load the raw data and run model for all animals 
if rerun:
    for iA,animal_id in enumerate(animal_ids): #for each animal

        animaldir   = os.path.join(rawdatadir,animal_id) 
        assert len(os.listdir(animaldir))==1, 'multiple stacks found per animal'
        sesdir      = os.path.join(animaldir,os.listdir(animaldir)[0])

        if animal_id in ['LPE12223', 'LPE12385']:
            dataV1 = get_stack_data_v2(os.path.join(sesdir,'STACK_V1'))
            dataPM = get_stack_data_v2(os.path.join(sesdir,'STACK_PM'))
        else: 
            dataV1 = get_stack_data_v1(os.path.join(sesdir,'STACK_V1'))
            dataPM = get_stack_data_v1(os.path.join(sesdir,'STACK_PM'))

        np.save(os.path.join(rawdatadir,'stackdata_%s.npy' % animal_id),(dataV1,dataPM))

#%% ################################################################
## Load previously processed stack data: 

# X=2 (fluo and cells), Y=number of slices 75, Z= number of animals 
dataV1          = np.zeros((2,nslices,nanimals)) 
dataPM          = np.zeros((2,nslices,nanimals)) 

for iA,animal_id in enumerate(animal_ids): #for each animal
    (dataV1[:,:,iA],dataPM[:,:,iA]) = np.load(os.path.join(rawdatadir,'stackdata_%s.npy' % animal_id))


#%% ######################  Make depth profile of # of labeled cells: ############################
clrs_areas  = get_clr_areas(['V1','PM'])

fig,axes   = plt.subplots(1,2,figsize=(6*cm,6*cm),sharey=True,sharex=False)
ax = axes[0]
plot_depthprofile(dataV1[1,:,:],ax,clrs_areas[0])
ax.set_title(r"$V1_{PM}$",c=clrs_areas[0])
ax.set_title(r"$V1$",c=clrs_areas[0])
ax.set_ylim([0,750])
ax.set_yticks(np.arange(0,800,step=250))
ax.invert_yaxis()
ax.set_ylabel('Cortical Depth')
ax.set_xlabel('tdTomato+ cells')

ax = axes[1]
plot_depthprofile(dataPM[1,:,:],ax,clrs_areas[1])
ax.set_title(r"$PM$",c=clrs_areas[1])

sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)
plt.tight_layout()
my_savefig(fig,savedir,'LabeledCells_Depth_%danimals' % nanimals)

#%% ######################  Make depth profile of fluorescence: ############################
fig,axes   = plt.subplots(1,2,figsize=(8*cm,8*cm),sharey=True,sharex=False)
ax = axes[0]
plot_depthprofile(minmax_scale(dataV1[0,:,:], feature_range=(0, 1), axis=0, copy=True),ax,clrs_areas[0])
ax.set_title(r"$V1_{PM}$",c=clrs_areas[0])
ax.set_title(r"$V1$",c=clrs_areas[0])
ax.set_xlabel('tdTomato fluorescence\n(norm.)')
ax.set_ylim([0,750])
ax.set_yticks(np.arange(0,800,step=250))
ax.invert_yaxis()
ax.set_ylabel('Cortical Depth')
ax = axes[1]
plot_depthprofile(minmax_scale(dataPM[0,:,:], feature_range=(0, 1), axis=0, copy=True),ax,clrs_areas[1])
ax.set_title(r"$PM$",c=clrs_areas[1])

sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)
plt.tight_layout()
my_savefig(fig,savedir,'Fluorescence_Depth_%danimals' % nanimals)



#%% ######################  Make depth profile of # of labeled cells: ############################
# Withcre and flp separate:
clrs_enzymes        = get_clr_recombinase(['cre','flp'])
clrs_enzymes = [sns.xkcd_rgb['orangered'],sns.xkcd_rgb['clear blue']]

fig,axes   = plt.subplots(1,2,figsize=(8*cm,8*cm),sharey=True,sharex=False)
ax = axes[0]
idx_V1cre = recomblist[np.isin(recomblist[:,0],animal_ids),1]=='cre'
# plot_depthprofile(minmax_scale(dataV1[1,:,idx_V1cre].T, feature_range=(0, 1), axis=0, copy=True),ax,clrs_enzymes[0])
# plot_depthprofile(minmax_scale(dataV1[1,:,~idx_V1cre].T, feature_range=(0, 1), axis=0, copy=True),ax,clrs_enzymes[1])
handles = []
handles.append(plot_depthprofile(dataV1[1,:,idx_V1cre].T,ax,clrs_enzymes[0]))
handles.append(plot_depthprofile(dataV1[1,:,~idx_V1cre].T,ax,clrs_enzymes[1]))
ax.legend(handles,['cre','flp'],frameon=False)
my_legend_strip(ax)
ax.set_title(r"$V1$",c='k')
ax.set_ylim([0,750])
ax.set_yticks(np.arange(0,800,step=250))
ax.invert_yaxis()
ax.set_ylabel('Cortical Depth')
ax.set_xlabel('tdTomato+ cells')

ax = axes[1]
idx_V1cre = recomblist[np.isin(recomblist[:,0],animal_ids),1]=='cre'
# plot_depthprofile(minmax_scale(dataPM[1,:,~idx_V1cre].T, feature_range=(0, 1), axis=0, copy=True),ax,clrs_enzymes[0])
# plot_depthprofile(minmax_scale(dataPM[1,:,idx_V1cre].T, feature_range=(0, 1), axis=0, copy=True),ax,clrs_enzymes[1])
plot_depthprofile(dataPM[1,:,~idx_V1cre].T,ax,clrs_enzymes[0])
plot_depthprofile(dataPM[1,:,idx_V1cre].T,ax,clrs_enzymes[1])
ax.set_title(r"$PM$",c='k')
# ax.legend(handles,['cre','flp'],frameon=False)
# my_legend_strip(ax)

sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)
plt.tight_layout()
my_savefig(fig,savedir,'LabeledCells_Depth_CreFlp_%danimals' % nanimals)

#%% ######################  Make depth profile of fluorescence: ############################
# Withcre and flp separate:

fig,axes   = plt.subplots(1,2,figsize=(8*cm,8*cm),sharey=True,sharex=False)
ax = axes[0]
idx_V1cre = recomblist[np.isin(recomblist[:,0],animal_ids),1]=='cre'
handles = []
handles.append(plot_depthprofile(minmax_scale(dataV1[0,:,idx_V1cre].T, feature_range=(0, 1), axis=0, copy=True),ax,clrs_enzymes[0]))
handles.append(plot_depthprofile(minmax_scale(dataV1[0,:,~idx_V1cre].T, feature_range=(0, 1), axis=0, copy=True),ax,clrs_enzymes[1]))
ax.legend(handles,['cre','flp'],frameon=False)
my_legend_strip(ax)
ax.set_title(r"$V1$",c='k')
ax.set_xlabel('tdTomato fluorescence\n(norm.)')
ax.set_ylim([0,750])
ax.set_yticks(np.arange(0,800,step=250))
ax.invert_yaxis()
ax.set_ylabel('Cortical Depth')

ax = axes[1]
idx_V1cre = recomblist[np.isin(recomblist[:,0],animal_ids),1]=='cre'
plot_depthprofile(minmax_scale(dataPM[0,:,~idx_V1cre].T, feature_range=(0, 1), axis=0, copy=True),ax,clrs_enzymes[0])
plot_depthprofile(minmax_scale(dataPM[0,:,idx_V1cre].T, feature_range=(0, 1), axis=0, copy=True),ax,clrs_enzymes[1])
# plot_depthprofile(minmax_scale(dataPM[0,:,:], feature_range=(0, 1), axis=0, copy=True),ax,clrs_areas[1])
ax.set_title(r"$PM$",c='k')

sns.despine(fig=fig, top=True, right=True, offset=1,trim=True)
plt.tight_layout()
my_savefig(fig,savedir,'Fluorescence_Depth_CreFlp_%danimals' % nanimals)




#%% ##### Show example planes alongside it:
exanimal            = 'LPE10192'
# exanimal            = 'LPE10885'
explanes_depths     = [70,180,320,450]
vmin                = 0
vmax                = 200

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,4,1)

handles = []
handles.append(plot_depthprofile(minmax_scale(dataV1[0,:,:], feature_range=(0, 1), axis=0, copy=True),ax1,clrs_areas[0]))
handles.append(plot_depthprofile(minmax_scale(dataPM[0,:,:], feature_range=(0, 1), axis=0, copy=True),ax1,clrs_areas[1]))
ax1.set_xlabel('Fluorescence')
ax1.set_title('Fluorescence')
ax1.set_ylim([0,750])
ax1.invert_yaxis()
ax1.set_ylabel('Cortical Depth')
for d in explanes_depths:
    ax1.axhline(d,color='k',linestyle=':',linewidth=1)
ax1.set_yticks(np.arange(0,800,step=100))

ax2 = fig.add_subplot(142)
handles = []
handles.append(plot_depthprofile(dataV1[1,:,:],ax2,clrs_areas[0]))
handles.append(plot_depthprofile(dataPM[1,:,:],ax2,clrs_areas[1]))
ax2.set_xlabel('#Labeled cells')
ax2.legend(handles,['V1','PM'],frameon=False)
ax2.set_title('#Labeled cells')
ax2.set_ylim([0,750])
ax2.invert_yaxis()
ax2.set_ylabel('')
ax2.legend(handles,['V1','PM'],frameon=False)
ax2.set_title('#Labeled cells')
ax2.set_yticks(np.arange(0,800,step=100))

for d in explanes_depths:
    ax2.axhline(d,color='k',linestyle=':',linewidth=1)

for i,d in enumerate(explanes_depths):
    ax = fig.add_subplot(4,4,i*4+3)

    direc_V1    = os.path.join(os.path.join(rawdatadir,exanimal),
                            os.listdir(os.path.join(rawdatadir,exanimal))[0],
                            'STACK_V1')
    fname       = os.listdir(direc_V1)[np.floor(d/10).astype('int32')] #get file name corresponding to ex slice depth
    
    reader              = imread(str(os.path.join(direc_V1,fname)))
    Data                = reader.data()
    imdata              = np.average(Data[1::2,:,:], axis=0)

    ax.imshow(imdata,cmap='gray',vmin=vmin,vmax=vmax)
    ax.text(0.1,0.9,u'%.0f \xb5m' % d,ha='left',va='top',color='w',fontsize=10)
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.add_patch(Rectangle((0,0),512,512,edgecolor = clrs_areas[0],
             linewidth = 3, fill=False))
    if i==0: 
         ax.set_title('V1')

    direc_PM    = os.path.join(os.path.join(rawdatadir,exanimal),
                            os.listdir(os.path.join(rawdatadir,exanimal))[0],
                            'STACK_PM')
    fname       = os.listdir(direc_PM)[np.floor(d/10).astype('int32')] 
    
    reader              = imread(str(os.path.join(direc_PM,fname)))
    Data                = reader.data()
    imdata              = np.average(Data[1::2,:,:], axis=0)

    ax = plt.subplot(4,4,i*4+4)
    ax.imshow(imdata,cmap='gray',vmin=vmin,vmax=vmax)
    ax.text(0.1,0.9,u'%.0f \xb5m' % d,ha='left',va='top',color='w',fontsize=10)
    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.add_patch(Rectangle((0,0),512,512,edgecolor =clrs_areas[1],
            linewidth = 3, fill=False))
    if i==0: 
         ax.set_title('PM')

plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3)

fig.savefig(os.path.join(savedir,'Labeling_Depth_%danimals_example_planes.png' % nanimals), 
            dpi=300,format = 'png',bbox_inches='tight')
fig.savefig(os.path.join(savedir,'Labeling_Depth_%danimals_example_planes.pdf' % nanimals),
            dpi=300,format = 'pdf',bbox_inches='tight')
